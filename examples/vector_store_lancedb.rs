// To run this example execute: cargo run --example vector_store_lancedb --features lancedb
// fastembed
#[cfg(feature = "lancedb")]
use langchain_rust::{
    schemas::Document,
    vectorstore::{lancedb::StoreBuilder, VecStoreOptions, VectorStore},
};
#[cfg(feature = "lancedb")]
use std::io::Write;

#[cfg(feature = "lancedb")]
#[tokio::main]
async fn main() {
    // Initialize Embedder

    use langchain_rust::embedding::FastEmbed;
    let embedder = FastEmbed::try_new().unwrap();

    let database_url = std::env::var("DATABASE_URL").unwrap_or("./tmp/lance".to_string());

    // Initialize the lancedb Vector Store
    let store = StoreBuilder::new()
        .embedder(embedder)
        .connection_url(database_url)
        .table("documents")
        .vector_dimensions(384)
        .build()
        .await
        .unwrap();

    // Initialize the tables in the database. This is required to be done only once.
    store.initialize().await.unwrap();

    // Add documents to the database
    let doc1 = Document::new(
        "langchain-rust is a port of the langchain python library to rust and was written in 2024.",
    );
    let doc2 = Document::new(
        "langchaingo is a port of the langchain python library to go language and was written in 2023."
    );
    let doc3 = Document::new(
        "Capital of United States of America (USA) is Washington D.C. and the capital of France is Paris."
    );
    let doc4 = Document::new("Capital of France is Paris.");

    store
        .add_documents(&vec![doc1, doc2, doc3, doc4], &VecStoreOptions::default())
        .await
        .unwrap();

    // Ask for user input
    print!("Query> ");
    std::io::stdout().flush().unwrap();
    let mut query = String::new();
    std::io::stdin().read_line(&mut query).unwrap();

    let results = store
        .similarity_search(&query, 10, &VecStoreOptions::default())
        .await
        .unwrap();

    if results.is_empty() {
        println!("No results found.");
        return;
    } else {
        results.iter().for_each(|r| {
            println!("Document: {}", r.page_content);
        });
    }
}
