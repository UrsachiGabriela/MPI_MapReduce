
from gensim.parsing.preprocessing import remove_stopwords
import string

from model import Tag


def remove_unnecessary_words(line):
    filtered_line = remove_stopwords(line)
    filtered_line = filtered_line.translate(str.maketrans(' ', ' ', string.punctuation))
    filtered_line = filtered_line.translate(str.maketrans(' ', ' ', string.digits))
    filtered_line = filtered_line.lower()
    filtered_line = [word for word in filtered_line.split() if len(word) > 2]

    return filtered_line

def create_path(tag, output_path, worker):
    if tag==Tag.MAP_OPERATION.value:
        return output_path + str(worker) + "/"
    return output_path + str(worker) +".txt"

def group_files(files, group_size):
    return  [files[i:i + group_size] for i in range(0, len(files), group_size)]


