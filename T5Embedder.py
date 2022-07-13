import torch
from transformers import T5EncoderModel, T5Tokenizer
import transformers

transformers.logging.set_verbosity_error()

class T5Embedder:
    def __init__(self, t5_dir, device):
        self.device = device

        transformer_link = "Rostlab/prot_t5_xl_uniref50"
        self.model = T5EncoderModel.from_pretrained(transformer_link, cache_dir=t5_dir, output_attentions=True)
        self.model = self.model.half()
        self.model = self.model.to(device)
        self.model.eval()

        self.vocab = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False, cache_dir=t5_dir )

    def get_embeddings(self, seq, detach=True, symmetry=True, APC=False):

        # TOP100 attention heads for detecting distances/contacts according to KW'S logistic regression analysis
        num_heads = 100
        contact_heads = [635, 755, 311, 271, 690, 761, 375, 617, 567, 23, 731, 535, 614, 640, 544,
                         335, 303, 678, 727, 15, 508, 239, 767, 732, 343, 28, 751, 604, 747, 495, 399,
                         702, 367, 725, 591, 407, 748, 661, 754, 759, 435, 247, 471, 16, 175, 697, 645,
                         613, 609, 670, 650, 713, 709, 13, 316, 728, 647, 215, 716, 663, 683, 1, 688, 527,
                         606, 764, 402, 597, 612, 124, 721, 143, 736, 564, 698, 24, 383, 599, 98, 207, 742,
                         439, 79, 111, 695, 722, 272, 97, 655, 699, 88, 516, 144, 156, 757, 625, 122, 284, 657, 109]

        seq = seq.replace('U','X').replace('Z','X').replace('O','X')
        seq_len = len(seq)
        seq = ' '.join(list(seq))

        token_encoding = self.vocab.batch_encode_plus([seq], add_special_tokens=True, padding="longest")
        input_ids      = torch.tensor(token_encoding['input_ids']).to(self.device)
        attention_mask = torch.tensor(token_encoding['attention_mask']).to(self.device)

        embedding_repr = self.model(input_ids, attention_mask=attention_mask)

        emb_1d = embedding_repr.last_hidden_state[0, :seq_len]

        # (24 x 32 x L x L) 24=n_layer; 32=n_heads
        emb_2d = torch.cat(embedding_repr[1], dim=0)[:,:,:seq_len,:seq_len]
        emb_2d = torch.reshape(emb_2d,(768,seq_len,seq_len)) # [contact_heads,:,:] # edit here for all heads

        if detach:
            emb_1d = emb_1d.detach()
            emb_2d = emb_2d.detach()

        # symmetry
        if symmetry:
            emb_2d = 0.5 * (emb_2d + torch.transpose(emb_2d, 1, 2))

        # APC
        if APC:
            for i in range(num_heads):
                diag = torch.diag(emb_2d[i,:,:])
                Fi = (torch.sum(emb_2d[i,:,:], dim=0) - diag) / seq_len
                Fj = (torch.sum(emb_2d[i,:,:], dim=1) - diag) / seq_len
                F = (torch.sum(emb_2d[i,:,:]) - torch.sum(diag)) / (seq_len*seq_len - seq_len)
                correction = torch.outer(Fi, Fj) / F
                emb_2d[i,:,:] -= correction

        return emb_1d, emb_2d
