package services;

import java.util.concurrent.CompletableFuture;

public interface UserStreamCommunication {

    CompletableFuture<Void> ask(String prompt);

}