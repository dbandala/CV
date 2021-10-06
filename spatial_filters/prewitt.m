% PREWITT SPATIAL FILTER
close all; clear all; clc;

current_directory = pwd;
img = imread([current_directory '\foto_tarea01.jpg']);
imshow(img);
title('Imagen original');
figure;

gray_level = rgb2gray(img);
imshow(gray_level);
title('Imagen en escala de grises');
figure;

f_transform = (double(gray_level));

h = fspecial('prewitt');
pre1 = uint8(round(filter2(h,gray_level)));
pre2 = fft(double(pre1));
imshow(pre1);
title('Imagen con filtro espacial Prewitt');
