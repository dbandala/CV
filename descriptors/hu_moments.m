% Hu moments
close all;clear all; clc; format short;
% get image and obtain gray scale version
current_directory = pwd;
img = imread([current_directory '\dogo.jpg']);
original = rgb2gray(img);
% modify geometry of original image
half = imresize(original,0.5);
rotated_180 = imrotate(original,180);
rotated_12 = imrotate(original,12);
rotated_45 = imrotate(original,45);
% show modified images
figure, imshow(original),title('Imagen original');
figure, imshow(half),title('Imagen a mitad del tamaño');
figure, imshow(rotated_180),title('Imagen rotada 180°');
figure, imshow(rotated_12),title('Imagen rotada 12°');
figure, imshow(rotated_45),title('Imagen rotada 45°');
% get geometric moments
original_mom = invmoments(original);
half_mom = invmoments(half);
rotated_180_mom = invmoments(rotated_180);
rotated_12_mom = invmoments(rotated_12);
rotated_45_mom = invmoments(rotated_45);
% get normalized moments
original_mom_normal = -sign(original_mom).*(log10(abs(original_mom)));
half_mom_normal = -sign(half_mom).*(log10(abs(half_mom)));
rotated_180_mom_normal = -sign(rotated_180_mom).*(log10(abs(rotated_180_mom)));
rotated_12_mom_normal = -sign(rotated_12_mom).*(log10(abs(rotated_12_mom)));
rotated_45_mom_normal = -sign(rotated_45_mom).*(log10(abs(rotated_45_mom)));
% generate result vector
results(1,:) = original_mom_normal;
results(2,:) = half_mom_normal;
results(3,:) = rotated_180_mom_normal;
results(4,:) = rotated_12_mom_normal;
results(5,:) = rotated_45_mom_normal;
meanResult = mean(results);
maxResult = max(results);
minResult = min(results);
dif = abs(maxResult-minResult);
% output results on console
file = fopen([current_directory '\results.csv'],'wt');
fprintf(file,'%4.6f , ',original_mom_normal); fprintf(file,'\n');
fprintf(file,'%4.6f , ',half_mom_normal); fprintf(file,'\n');
fprintf(file,'%4.6f , ',rotated_180_mom_normal); fprintf(file,'\n');
fprintf(file,'%4.6f , ',rotated_12_mom_normal); fprintf(file,'\n');
fprintf(file,'%4.6f , ',rotated_45_mom_normal); fprintf(file,'\n');
fprintf(file,'%4.6f , ',dif); fclose(file);
