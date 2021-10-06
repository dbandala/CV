clc
clear all
close all
I=imread('im1.png');
imshow(I)
hold on
h = drawfreehand;
hold off

 I=im2double(I);
  x=h.Position(:,2);
y=h.Position(:,1);  
P=[x(:) y(:)];
  Options=struct;
  Options.Verbose=true;
  Options.Iterations=600;
  Options.Wedge=2;
  Options.Wline=0;
  Options.Wterm=0;
  Options.Kappa=4;
  Options.Sigma1=8;
  Options.Sigma2=8;
  Options.Alpha=0.1;
  Options.Beta=0.1;
  Options.Mu=0.2;
  Options.Delta=-0.1;
  Options.GIterations=600;
  [O,J]=Snake2D(I,P,Options);
 