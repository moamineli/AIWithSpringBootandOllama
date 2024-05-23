package services;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.TokenStream;

public interface ModelCommunication {

	String[] x = {
			"ton identité c'est un agent de XXX qui donne des réponses courtes liés au texte donné seulement et en francais seulement",

	};

	default String[] getX() {
		return x;
	}

	@SystemMessage(value = {
			"ton identité c'est un agent de XXX qui donne des réponses courtes liés au texte donné seulement et en francais seulement",

	})
	TokenStream chatWithModel(String message);

}