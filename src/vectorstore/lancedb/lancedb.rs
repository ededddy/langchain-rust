use std::{error::Error, sync::Arc};

use crate::{embedding::Embedder, schemas::Document, vectorstore::VectorStore};
use arrow_array::{Float64Array, RecordBatchIterator};
use arrow_schema::FieldRef;
use async_trait::async_trait;
use futures::TryStreamExt;
use lancedb::{
    arrow::arrow_schema::Schema,
    index::Index,
    query::{ExecutableQuery, QueryBase},
    Connection, DistanceType,
};
use serde::{Deserialize, Serialize};
use serde_arrow::schema::{SchemaLike, TracingOptions};
use serde_json::json;
use uuid::Uuid;

pub struct Store {
    pub(crate) connection: Connection,
    pub(crate) table: String,
    pub(crate) vector_dimensions: i32,
    pub(crate) embedder: Arc<dyn Embedder>,
}

#[derive(Serialize, Deserialize)]
pub struct VectorRecord {
    pub id: String,
    pub text: String,
    pub metadata: String,
    pub text_embedding: Vec<f64>,
}

impl Store {
    pub async fn initialize(&self) -> Result<(), Box<dyn Error>> {
        self.create_table_if_not_exists().await?;
        Ok(())
    }

    async fn create_table_if_not_exists(&self) -> Result<(), Box<dyn Error>> {
        let schema = Vec::<FieldRef>::from_type::<VectorRecord>(TracingOptions::default())?;
        let tb = self
            .connection
            .create_empty_table(&self.table, Arc::new(Schema::new(schema)))
            .execute()
            .await;
        match tb {
            Ok(table) => {
                table
                    .create_index(&["text_embedding"], Index::Auto)
                    .execute()
                    .await?
            }
            Err(error) => match error {
                lancedb::Error::TableAlreadyExists { name } => {
                    eprintln!("lancedb : table `{name}` already exists");
                    return Ok(());
                }
                _ => return Err(Box::new(error)),
            },
        }

        Ok(())
    }

    async fn drop_table(&self) -> Result<(), Box<dyn Error>> {
        let tables: Vec<String> = self
            .connection
            .table_names()
            .execute()
            .await?
            .into_iter()
            .collect();
        if !tables.contains(&self.table) {
            if let Err(error) = self.connection.drop_table(&self.table).await {
                return Err::<(), Box<dyn Error>>(Box::new(error));
            }
        }
        Ok(())
    }
}

#[async_trait]
impl VectorStore for Store {
    async fn add_documents(
        &self,
        docs: &[crate::schemas::Document],
        opt: &crate::vectorstore::VecStoreOptions,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let texts: Vec<String> = docs.iter().map(|d| d.page_content.clone()).collect();

        let embedder = opt.embedder.as_ref().unwrap_or(&self.embedder);

        let vectors = embedder.embed_documents(&texts).await?;
        if vectors.len() != docs.len() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Number of vectors and documents do not match",
            )));
        }

        let schema = Vec::<FieldRef>::from_type::<VectorRecord>(TracingOptions::default())?;
        let vector_records: Vec<VectorRecord> = docs
            .into_iter()
            .zip(vectors)
            .map(|(d, v)| VectorRecord {
                id: Uuid::new_v4().to_string(),
                text: d.page_content.clone(),
                metadata: json!(d.metadata).to_string(),
                text_embedding: v,
            })
            .collect();

        let table = &self.table;
        let tb = self.connection.open_table(table).execute().await?;

        let batches = RecordBatchIterator::new(
            vec![serde_arrow::to_record_batch(&schema, &vector_records)?]
                .into_iter()
                .map(Ok),
            tb.schema().await?.clone(),
        );

        let ids: Vec<String> = vector_records.into_iter().map(|v| v.id.clone()).collect();

        match tb.add(batches).execute().await {
            Ok(_) => Ok(ids),
            Err(error) => Err(Box::new(error)),
        }
    }

    async fn similarity_search(
        &self,
        query: &str,
        limit: usize,
        _opt: &crate::vectorstore::VecStoreOptions,
    ) -> Result<Vec<crate::schemas::Document>, Box<dyn std::error::Error>> {
        let query_vector = self.embedder.embed_query(query).await?;
        let table = self.connection.open_table(&self.table).execute().await?;
        let results = table
            .query()
            .nearest_to(query_vector)
            .unwrap()
            .column("text_embeddings")
            .distance_type(DistanceType::Cosine)
            .limit(limit)
            .execute()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let mut items: Vec<Document> = Vec::with_capacity(results.len());
        for result in results {
            let scores: Vec<f64> = result
                .column_by_name("_distance")
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .values()
                .to_vec();
            let vec_record: Vec<VectorRecord> = serde_arrow::from_record_batch(&result)?;
            vec_record.into_iter().enumerate().for_each(|(i, vr)| {
                items.push(Document {
                    page_content: vr.text,
                    metadata: serde_json::from_str(&vr.metadata).unwrap(),
                    score: scores[i],
                })
            })
        }
        Ok(items)
    }
}
