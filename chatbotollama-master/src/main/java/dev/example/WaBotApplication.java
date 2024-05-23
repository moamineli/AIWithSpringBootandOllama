package dev.example;

import java.io.IOException;
import java.util.List;
import java.util.Scanner;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.ApplicationRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.core.io.ResourceLoader;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiTokenizer;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import services.ChatService;

@SpringBootApplication
@ComponentScan(basePackages = "services")
public class WaBotApplication {

	/**
	 * Run WaBotApplicationTest to see simulated conversation with customer support
	 * agent
	 */

	@Value("${model.tokenizer}")
	private String modelTokenizer;

	@Value("${chat.history.size}")
	private Integer chatHistorySize;

	@Value("${bc.filepath}")
	private String bcFilePath;
	
	
	@Value("${ollama.model}")
	private String ollamaModel;
	
	@Value("${ollama.baseuri}")
	private String baseUri;
	
	@Value("${embedding.tokensplit}")
	private Integer tokenSplit;
	
	@Value("${embedding.tokenseacrhscore}")
	private double minScore;
	
	@Value("${embedding.maxSearchResults}")
	private int maxSearchResults;
	
	@Value("${ollama.model.temperature}")
	private double temperature;

	@Bean
	ApplicationRunner interactiveChatRunner(ContentRetriever content) {
		return args -> {

			Scanner scanner = new Scanner(System.in);
			ChatService chatService = new ChatService(baseUri, ollamaModel, content,tokenSplit,minScore,maxSearchResults,temperature);
			while (true) {
				System.out.print("Type 'exit' to exit the program \n");
				String userPrompt = scanner.nextLine();
				if (userPrompt.equals("exit")) {
					break;
				}
				// Change to streaming model
				chatService.chatWithModel(userPrompt);
				chatService.ask(userPrompt).join();
			}
			scanner.close();
		};
	}

	@Bean
	ContentRetriever contentRetriever(EmbeddingStore<TextSegment> embeddingStore, EmbeddingModel embeddingModel) {
		return EmbeddingStoreContentRetriever.builder().embeddingStore(embeddingStore).embeddingModel(embeddingModel)
				.maxResults(maxSearchResults).minScore(minScore).build();
	}

	@Bean
	EmbeddingModel embeddingModel() {
		return new AllMiniLmL6V2EmbeddingModel();
	}

	@Bean
	EmbeddingStore<TextSegment> embeddingStore(EmbeddingModel embeddingModel, ResourceLoader resourceLoader)
			throws IOException {

		// 1. Create an in-memory embedding store
		EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

		//Load BC
		List<Document> documents = FileSystemDocumentLoader.loadDocuments(bcFilePath, new TextDocumentParser());

		// 3. Split the document into segments 000 tokens each
		// 4. Convert segments into embeddings
		// 5. Store embeddings into embedding store
		// All this can be done manually, but we will use EmbeddingStoreIngestor to
		// automate this:
		DocumentSplitter documentSplitter = DocumentSplitters.recursive(tokenSplit, 0, new OpenAiTokenizer("gpt-4"));
		EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder().documentSplitter(documentSplitter)
				.embeddingModel(embeddingModel).embeddingStore(embeddingStore).build();
		ingestor.ingest(documents);

		return embeddingStore;
	}

	public static void main(String[] args) {
		SpringApplication.run(WaBotApplication.class, args);
	}
}
