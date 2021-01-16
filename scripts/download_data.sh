# Get data online
if [ -d "data" ]
then
    echo "Data directory already exists. Do nothing!"
else
    mkdir "data"
    cd data
    wget https://people.csail.mit.edu/yujia/files/r2a/data.zip
    unzip data.zip      
    mkdir hotel_review
    cd hotel_review
    mkdir hotel0 hotel1 hotel2 annoated
    #text data
    cp ../data/Orcale/hotel_Location.train hotel0/train.tsv
    cp ../data/Orcale/hotel_Location.dev hotel0/dev.tsv
    cp ../data/Orcale/hotel_Service.train hotel1/train.tsv
    cp ../data/Orcale/hotel_Service.dev hotel1/dev.tsv
    cp ../data/Orcale/hotel_Cleanliness.train hotel2/train.tsv
    cp ../data/Orcale/hotel_Cleanliness.dev hotel2/dev.tsv
    #annoated data
    cp  ../data/target/hotel_Cleanliness.train annoated/hotel_Cleanliness.train
    cp ../data/target/hotel_Location.train annoated/hotel_Location.train 
    cp ../data/target/hotel_Service.train annoated/hotel_Service.train 
fi

# Get the embedding
if [ -d "embeddings" ]
then
    echo "Embeddings directory already exists. Do nothing!"
else
    mkdir "embeddings"
    # get embeddings
    cd embeddings
    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip    
fi

