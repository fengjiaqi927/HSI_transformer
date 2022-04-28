clear all
close all
clc

load data_Indian_pines
% (43, 21, 11)
r = imageCube(:,:,43);
r = (r-min(min(r)))/(max(max(r))-min(min(r)));
figure();
imshow(r)
g = imageCube(:,:,21);
figure();
g = (g-min(min(g)))/(max(max(g))-min(min(g)));
imshow(g)
b = imageCube(:,:,11);
figure();
b = (b-min(min(b)))/(max(max(b))-min(min(b)));
imshow(b)
RGB(:,:,1)=r;
RGB(:,:,2)=g;
RGB(:,:,3)=b;
figure();
imshow(RGB,[])
imwrite(RGB,'rgb.bmp');


figure()
imshow(labeloverlay(RGB,boundarymask(superpixels(RGB,200)),'Transparency',0,'Colormap',[1,1,1]))
load('IndianPines_GT.mat')

figure()
imshow(labeloverlay(label2rgb(groundTruth),boundarymask(superpixels(RGB,200)),'Transparency',0,'Colormap',[0,0,0]))
segmentation_results = int32(superpixels(RGB,200));
imwrite(uint8(segmentation_results),'sr.bmp');
save('segmentation_results.mat','segmentation_results')

figure()
imshow(labeloverlay(imresize(label2rgb(groundTruth),[145*2,145*2]),boundarymask(superpixels(imresize(RGB,[145*2,145*2]),200)),'Transparency',0,'Colormap',[0.3,0.3,0.3]))


figure()
imshow(imresize(label2rgb(groundTruth),[145*3,145*3]))

figure()
imshow(imresize(boundarymask(superpixels(RGB,200)),[145*3,145*3]))

figure()
imshow(boundarymask(superpixels(imresize(RGB,[145*2,145*2]),200)))

figure()
imshow(superpixels(RGB,200),[])