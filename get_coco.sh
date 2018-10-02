mkdir coco
cd coco
mkdir images
cd images

wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/zips/test2015.zip

unzip train2014.zip
unzip val2014.zip
unzip test2015.zip

rm train2014.zip
rm val2014.zip
rm test2015.zip

cd ../
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2014.zip
wget http://images.cocodataset.org/annotations/image_info_test2015.zip

unzip annotations_trainval2014.zip
unzip stuff_annotations_trainval2014.zip
unzip image_info_test2015.zip

rm annotations_trainval2014.zip
rm stuff_annotations_trainval2014.zip
rm image_info_test2015.zip
