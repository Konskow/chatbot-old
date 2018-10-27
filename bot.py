import torch

from models.decoder import Decoder
from models.encoder import Encoder
from models.greedy_search_decoder import GreedySearchDecoder
from prepare_data import normalize_string, MAX_LENGTH
from train_utils import indices_from_sentence


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH, device='cpu'):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indices_from_sentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.tensor(indexes_batch).transpose(0, 1).to(dtype=torch.long)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length, vocabulary)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluate_input(encoder, decoder, searcher, vocabulary):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalize_string(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, vocabulary, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


if __name__ == "__main__":
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

    vocabulary = {}
    model_params = {}

    load_filename = 'temp/chatbot/corpus/2-2_500/1000_checkpoint.tar'
    checkpoint = torch.load(load_filename)
    # If loading a model trained on GPU to CPU
    # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embeddings']
    vocabulary = checkpoint['vocabulary']
    model_params = checkpoint['model_params']

    embeddings = torch.nn.Embedding(vocabulary.num_words, model_params['hidden_size'])
    if load_filename:
        embeddings.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = Encoder(model_params, embeddings)
    decoder = Decoder(model_params, embeddings, vocabulary.num_words)
    if load_filename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)

    evaluate_input(encoder, decoder, searcher, vocabulary)
