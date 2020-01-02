#set -x
set -e

input_path=${1}

java -cp "tools/*" -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP\
    -annotators tokenize,ssplit\
    -file ${input_path}\
    -outputFormat conll\
    -output.columns word\
    -output.prettyPrint false\
    -outputDirectory /tmp/

