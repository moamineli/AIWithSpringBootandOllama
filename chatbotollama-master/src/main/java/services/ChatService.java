package services;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.time.Duration;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.concurrent.CompletableFuture;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Autowired;

import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.StreamingChatLanguageModel;
import dev.langchain4j.model.ollama.OllamaStreamingChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.TokenStream;

@Service
public class ChatService implements UserStreamCommunication, ModelCommunication {

	private final String ollamaModel;
	private final String ollamaBaseUri;
	private final Integer tokenSplit;
	private final double minScore;
	private final int maxSearchResults;
	private final double temperature;

	private final StreamingChatLanguageModel languageModel;
	private final ModelCommunication assistant;

	private final LocalDateTime currentTime = LocalDateTime.now();
	private final DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMdd_HHmm");

	@Autowired
	public ChatService(
			@Value("${ollama.baseuri}") String ollamaBaseUri,
			@Value("${ollama.model}") String ollamaModel,
			ContentRetriever contentRetriever,
			@Value("${embedding.tokensplit}") Integer tokenSplit,
			@Value("${embedding.tokenseacrhscore}") double minScore,
			@Value("${embedding.maxSearchResults}") int maxSearchResults,
			@Value("${ollama.model.temperature}") double temperature) {

		this.ollamaModel = ollamaModel;
		this.ollamaBaseUri = ollamaBaseUri;
		this.tokenSplit = tokenSplit;
		this.minScore = minScore;
		this.maxSearchResults = maxSearchResults;
		this.temperature = temperature;
		this.languageModel = connectModel(ollamaBaseUri, ollamaModel, temperature);

		// Memorize for a specified number of messages continuously
		ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(5);  // Adjusted to chat.history.size if needed
		this.assistant = AiServices.builder(ModelCommunication.class)
				.contentRetriever(contentRetriever)
				.streamingChatLanguageModel(this.languageModel)
				.chatMemory(chatMemory)
				.build();
	}

	public String getFileName() {
		return currentTime.format(formatter) + "_mo_" + ollamaModel + "_to_" + tokenSplit + "_te" + temperature
				+ "_tss" + minScore + "_msr" + maxSearchResults + ".txt";
	}

	public CompletableFuture<Void> ask(String userPrompt) {
		TokenStream tokenStream = chatWithModel(userPrompt);
		CompletableFuture<Void> future = new CompletableFuture<>();
		tokenStream.onNext(System.out::print).onComplete(k -> {
			System.out.println();
			writeToFile(userPrompt, k.toString()); // Write to file
			future.complete(null);
		}).onError(Throwable::printStackTrace).start();
		return future;
	}

	private void writeToFile(String question, String response) {
		try (BufferedWriter writer = new BufferedWriter(new FileWriter(getFileName(), true))) {
			writer.write("User: " + question + "\n");
			writer.write("Model: " + response + "\n\n");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public TokenStream chatWithModel(String message) {
		return assistant.chatWithModel(message);
	}

	private static StreamingChatLanguageModel connectModel(String url, String modelName, double temperature) {
		return OllamaStreamingChatModel.builder()
				.baseUrl(url)
				.modelName(modelName)
				.temperature(temperature)
				.timeout(Duration.ofMinutes(1))
				.build();
	}
}
