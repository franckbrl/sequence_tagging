import sys

from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config


def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned


def get_sents(filename):
    sent = []
    for line in open(filename, 'r'):
        if line == "\n":
            yield sent
            sent = []
            continue
        word = line.split()[0]
        sent.append(word)

        
def label_file(model, filename, filename_out):
    """label a file in conll format (vertical)

    Args:
        model: instance of NERModel
        filename: file to label
        filename_out: file where to write result

    """
    outfile = open(filename_out, "w")
    for words_raw in get_sents(filename):
        #print(words_raw)
        preds = model.predict(words_raw)
        #preds = words_raw

        for word, pred in zip(words_raw, preds):
            outfile.write(word + '\t' + pred + '\n')
        outfile.write('\n')
        
        #for key, seq in to_print.items():
        #    model.logger.info(seq)




def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # create dataset
    # test  = CoNLLDataset(config.filename_test, config.processing_word,
    #                      config.processing_tag, config.max_iter)

    # evaluate and interact
    #model.evaluate(test)
    #interactive_shell(model)
    label_file(model, sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
