wget -N "http://www.robots.ox.ac.uk/~arnabg/cartoonset10k.zip"

unzip cartoonset10k.zip
rm cartoonset10k.zip
mkdir ../data

cp -r cartoonset10k ../data
rm -rf cartoonset10k
echo "Wait till the autocomplete data gets generated"
python autocomplete/data_generation.py --scribble_dir '../data/cartoonset10k/scribbles/'

mkdir ../data/autocomplete-cartoons
mkdir ../data/autocomplete-cartoons/images
mkdir ../data/autocomplete-cartoons/scribbles

cp -r ../data/cartoonset10k/scribbles/cartoon ../data/autocomplete-cartoons/images/

cp -r autocomplete/cartoon ../data/autocomplete-cartoons/scribbles/

