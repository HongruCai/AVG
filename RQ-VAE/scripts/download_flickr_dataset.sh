wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part00"
wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part01"
wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part02"
cat flickr30k_part00 flickr30k_part01 flickr30k_part02 > flickr30k.zip
rm flickr30k_part00 flickr30k_part01 flickr30k_part02
unzip -q flickr30k.zip -d ./flickr30k
rm flickr30k.zip
echo "Downloaded Flickr30k dataset successfully."