use std::{error::Error, sync::Arc};

use super::Store;
use lancedb::{connect, Connection};

use crate::embedding::embedder_trait::Embedder;

pub struct StoreBuilder {
    connection: Option<Connection>,
    connection_url: Option<String>,
    table: String,
    vector_dimensions: i32,
    embedder: Option<Arc<dyn Embedder>>,
}

impl Default for StoreBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl StoreBuilder {
    pub fn new() -> Self {
        StoreBuilder {
            connection: None,
            connection_url: None,
            table: "documents".to_string(),
            vector_dimensions: 0,
            embedder: None,
        }
    }

    pub fn connection(mut self, connection: Connection) -> Self {
        self.connection = Some(connection);
        self.connection_url = None;
        self
    }

    pub fn connection_url<S: Into<String>>(mut self, connection_url: S) -> Self {
        self.connection_url = Some(connection_url.into());
        self.connection = None;
        self
    }

    pub fn table(mut self, table: &str) -> Self {
        self.table = table.into();
        self
    }

    pub fn vector_dimensions(mut self, vector_dimensions: i32) -> Self {
        self.vector_dimensions = vector_dimensions;
        self
    }

    pub fn embedder<E: Embedder + 'static>(mut self, embedder: E) -> Self {
        self.embedder = Some(Arc::new(embedder));
        self
    }

    // Finalize the builder and construct the Store object
    pub async fn build(self) -> Result<Store, Box<dyn Error>> {
        if self.embedder.is_none() {
            return Err("Embedder is required".into());
        }
        let mut connection_str = self.connection_url.clone();
        let mut connection = self.connection.clone();

        if connection_str.is_none() {
            connection_str = Some("./tmp/tmp_lancedb".into());
        }

        if connection.is_none() {
            connection = Some(connect(&connection_str.unwrap()).execute().await?)
        }

        Ok(Store {
            connection: connection.unwrap(),
            table: self.table,
            vector_dimensions: self.vector_dimensions,
            embedder: self.embedder.unwrap(),
        })
    }
}
