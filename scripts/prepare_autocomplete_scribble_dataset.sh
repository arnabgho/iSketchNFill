wget -N "http://www.robots.ox.ac.uk/~arnabg/scribble_dataset.zip"

unzip scribble_dataset.zip
rm scribble_dataset.zip
mkdir ../data

cp -r scribble_dataset ../data
rm -rf scribble_dataset
echo "Wait till the autocomplete data gets generated"
python autocomplete/data_generation.py --scribble_dir '../data/scribble_dataset/scribbles/'

mkdir ../data/autocomplete-scribble-dataset
mkdir ../data/autocomplete-scribble-dataset/images
mkdir ../data/autocomplete-scribble-dataset/scribbles

cp -r ../data/scribble_dataset/scribbles/basketball ../data/autocomplete-scribble-dataset/images/
cp -r ../data/scribble_dataset/scribbles/chicken ../data/autocomplete-scribble-dataset/images/
cp -r ../data/scribble_dataset/scribbles/cookie ../data/autocomplete-scribble-dataset/images/
cp -r ../data/scribble_dataset/scribbles/cupcake ../data/autocomplete-scribble-dataset/images/
cp -r ../data/scribble_dataset/scribbles/moon ../data/autocomplete-scribble-dataset/images/
cp -r ../data/scribble_dataset/scribbles/orange ../data/autocomplete-scribble-dataset/images/
cp -r ../data/scribble_dataset/scribbles/pineapple ../data/autocomplete-scribble-dataset/images/
cp -r ../data/scribble_dataset/scribbles/soccer ../data/autocomplete-scribble-dataset/images/
cp -r ../data/scribble_dataset/scribbles/strawberry ../data/autocomplete-scribble-dataset/images/
cp -r ../data/scribble_dataset/scribbles/watermelon ../data/autocomplete-scribble-dataset/images/

cp -r autocomplete/basketball ../data/autocomplete-scribble-dataset/scribbles/
cp -r autocomplete/chicken ../data/autocomplete-scribble-dataset/scribbles/
cp -r autocomplete/cookie ../data/autocomplete-scribble-dataset/scribbles/
cp -r autocomplete/cupcake ../data/autocomplete-scribble-dataset/scribbles/
cp -r autocomplete/moon ../data/autocomplete-scribble-dataset/scribbles/
cp -r autocomplete/orange ../data/autocomplete-scribble-dataset/scribbles/
cp -r autocomplete/pineapple ../data/autocomplete-scribble-dataset/scribbles/
cp -r autocomplete/soccer ../data/autocomplete-scribble-dataset/scribbles/
cp -r autocomplete/strawberry ../data/autocomplete-scribble-dataset/scribbles/
cp -r autocomplete/watermelon ../data/autocomplete-scribble-dataset/scribbles/
