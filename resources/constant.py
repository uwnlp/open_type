from collections import namedtuple, defaultdict

def load_vocab_dict(vocab_file_name, vocab_max_size=None, start_vocab_count=None):
  with open(vocab_file_name) as f:
    text = [x.strip() for x in f.readlines()]
    if vocab_max_size:
      text = text[:vocab_max_size]
    if start_vocab_count:
      file_content = dict(zip(text, range(0 + start_vocab_count, len(text) + start_vocab_count)))
    else:
      file_content = dict(zip(text, range(0, len(text))))
  return file_content

FILE_ROOT = '/home/eunsol/Data/entity_type_data/release/'
GLOVE_VEC = '/home/eunsol/Data/pretrainedVector/glove.840B.300d.txt'
EXP_ROOT = '/home/eunsol/Experiment/pytorch/'

ANSWER_NUM_DICT = {"open": 10331, "onto":89, "wiki": 4600, "kb":130, "gen":9}

KB_VOCAB = load_vocab_dict(FILE_ROOT + "/ontology/types.txt", 130)
WIKI_VOCAB = load_vocab_dict(FILE_ROOT + "/ontology/types.txt", 4600)
ANSWER_VOCAB = load_vocab_dict(FILE_ROOT + "/ontology/types.txt")
ONTO_ANS_VOCAB = load_vocab_dict(FILE_ROOT + '/ontology/onto_ontology.txt')
ANS2ID_DICT = {"open": ANSWER_VOCAB, "wiki": WIKI_VOCAB, "kb": KB_VOCAB, "onto":ONTO_ANS_VOCAB}

open_id2ans = {v: k for k, v in ANSWER_VOCAB.items()}
wiki_id2ans = {v: k for k, v in WIKI_VOCAB.items()}
kb_id2ans = {v:k for k,v in KB_VOCAB.items()}
g_id2ans = {v: k for k, v in ONTO_ANS_VOCAB.items()}

ID2ANS_DICT = {"open": open_id2ans, "wiki": wiki_id2ans, "kb": kb_id2ans, "onto":g_id2ans}
label_string = namedtuple("label_types", ["head", "wiki", "kb"])
LABEL = label_string("HEAD", "WIKI", "KB")

CHAR_DICT = defaultdict(int)
char_vocab = [u"<unk>"]
with open(FILE_ROOT + "/ontology/char_vocab.english.txt") as f:
  char_vocab.extend(c.strip() for c in f.readlines())
  CHAR_DICT.update({c: i for i, c in enumerate(char_vocab)})
