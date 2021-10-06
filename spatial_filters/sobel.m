% SOBEL SPATIAL FILTER
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

s = fspecial('sobel');
sob = uint8(round(filter2(s,gray_level)));
imshow(sob);
title('Imagen con filtro espacial Sobel');
