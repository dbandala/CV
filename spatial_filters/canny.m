% CANNY SPATIAL FILTER
close all;clear all; clc;

figure;
current_directory = pwd;
img = imread([current_directory '\foto_tarea01.jpg']);
imshow(img);
title('Imagen original');

figure;
gray_level = rgb2gray(img);
imshow(gray_level);
title('Imagen en escala de grises');

figure;
can = edge(gray_level,'canny',[0 1/5]);
imshow(can);
title('Imagen con filtro espacial Canny');
