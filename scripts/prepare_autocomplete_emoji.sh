wget -N "http://www.robots.ox.ac.uk/~arnabg/cartoonset10k.zip"

unzip cartoonset10k.zip
rm cartoonset10k.zip
mkdir ../data

mv cartoonset10k ../data
rm -rf cartoonset10k
echo "Wait till the autocomplete data gets generated"
python autocomplete/data_generation.py

mkdir ../data/autocomplete-cartoons
mkdir ../data/autocomplete-cartoons/images
mkdir ../data/autocomplete-cartoons/scribbles

mv ../data/cartoonset10k/scribbles/cartoon ../data/autocomplete-cartoons/images/

mv autocomplete/cartoon ../data/autocomplete-cartoons/scribbles/

