clc
clear all

close all


I = im2double(imread('medtest.png'));

%imshow(I)
%%

 x=177; y=321;
 
 J = regiongrowing(I,x,y,0.3); 
 figure, imshow(I+J);