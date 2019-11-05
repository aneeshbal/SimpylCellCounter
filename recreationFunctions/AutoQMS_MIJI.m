8-javaaddpath 'C:\Program Files\MATLAB\R2017b\java\ij.jar'
javaaddpath 'C:\Program Files\MATLAB\R2017b\java\mij.jar'
addpath 'C:\Users\anees\Desktop\Fiji.app\scripts'
Miji(false);
%% Choudhry 2016 - recreate main results
inter = ij.macro.Interpreter;
inter.batchMode = true;

images = dir('*.jpg');
ds = {images.name}; ds = ds';
numfiles = numel(ds);
folder = {images.folder}; folder = folder';
t_array = zeros(numel(numfiles),1);

for jj = 1:numfiles
    tic
    open = 'Open...'; 
    %comb_array = 'C:\Users\anees\Desktop\SimpylCellCounter\trials\dapi_fos.jpg';
    comb_array = fullfile(folder{jj,1},ds{jj,1});
    dd = ['path=[',comb_array,']'];
    img = imread(comb_array); img = img(:,:,1); sz = size(img);
    img = reshape(img, [sz(1)*sz(2),1]);
    th = mean(img)*0.7;
    %th = 115;
    
    %open_paths = 'path=[comb_array]';
    MIJ.run(open,dd);
    MIJ.run("Sharpen");
    MIJ.run("Enhance Contrast...", "saturated=0.001");
    MIJ.run("Remove Outliers...", "radius=0 threshold=0 show=Outline which=Dark");
    MIJ.run("8-bit");
    %MIJ.run("Threshold...")
    MIJ.setThreshold(th, 255);
    MIJ.run("Convert to Mask");
    MIJ.run("Convert to Mask");
    MIJ.run("Find Edges");
    MIJ.run("Gaussian Blur...", "sigma=2");
    %setOption("BlackBackground", false);
    MIJ.run("Make Binary");
    MIJ.run("Close-");
    MIJ.run("Fill Holes");
    MIJ.run("Remove Outliers...", "radius=8 threshold=0 which=Dark");
    MIJ.run("Maximum...", "radius=2");
    MIJ.run("Close-");
    MIJ.run("Fill Holes");
    MIJ.run("Minimum...", "radius=3");
    MIJ.run("Despeckle");
    MIJ.run("Watershed");
    MIJ.run("Remove Outliers...", "radius=12 threshold=0 which=Dark");
    MIJ.run("Analyze Particles...", "size=1000-1500 circularity=0.8-1.00 show=Outlines summarize");
    %IJ.run("Overlay Options...", "stroke=red width=2 set apply");
    %Overlay.copy;
    %MIJ.run(open,dd);
    %MIJ.newImage("Untitled", "8-bit black", 256, 256, 1);
    %MIJ.Overlay.paste;
    %MIJ.run("Analyze Particles...", "summarize")
    t = toc;
    t_array(jj,1) = t;
end

%%
inter.batchMode = false;
images = dir('*.jpg');
ds = {images.name}; ds = ds';
numfiles = numel(ds);
folder = {images.folder}; folder = folder';

for jj = 1:numfiles
    open = 'Open...'; 
    comb_array = fullfile(folder{jj,1},ds{jj,1});
    dd = ['path=[',comb_array,']'];
    %open_paths = 'path=[comb_array]';
    MIJ.run(open,dd);
    MIJ.run("8-bit");
    MIJ.setThreshold(115,120);
    %MIJ.setTool("multipoint");
    
end
inter.batchMode = true;

%%
inter.batchMode = false;
images = dir('*.jpg');
ds = {images.name}; ds = ds';
numfiles = numel(ds);
folder = {images.folder}; folder = folder';

for jj = 1:numfiles
    open = 'Open...'; 
    comb_array = fullfile(folder{jj,1},ds{jj,1});
    dd = ['path=[',comb_array,']'];
    %open_paths = 'path=[comb_array]';
    MIJ.run(open,dd);
    MIJ.run("Color Threshold...");
    %MIJ.setThreshold(115,120);
    %MIJ.setTool("multipoint");
    
end
inter.batchMode = true;

%%
%inter.batchMode = false;
%inter = ij.macro.Interpreter;
inter.batchMode = false;

images = dir('*.jpg');
ds = {images.name}; ds = ds';
numfiles = numel(ds);
folder = {images.folder}; folder = folder';

for jj = 1:50
    open = 'Open...'; 
    comb_array = fullfile(folder{jj,1},ds{jj,1});
    dd = ['path=[',comb_array,']'];
    MIJ.run(open,dd);
    MIJ.run("8-bit");
    MIJ.setThreshold(100,255);
    MIJ.run("Convert to Mask");
    MIJ.run("Convert to Mask");   
    %MIJ.run("Subtract Background...", "rolling=2 light create");
    MIJ.run("Convert to Mask");
    MIJ.run("Analyze Particles...", "summarize");
    %clear java
    %MIJ.setTool("multipoint");
    
end
%inter.batchMode = true;

%% create watershed training masks

inter = ij.macro.Interpreter;
inter.batchMode = true;

images = dir('*.jpg');
ds = {images.name}; ds = ds';
numfiles = numel(ds);
folder = {images.folder}; folder = folder';

for jj = 1:1000


    open = 'Open...'; 
    comb_array = fullfile(folder{jj,1},ds{jj,1});
    dd = ['path=[',comb_array,']'];
    MIJ.run(open,dd);
    MIJ.run("Convert to Mask");   
    %MIJ.run("Subtract Background...", "rolling=2 light create");
    MIJ.run("Watershed");
    %MIJ.run("Analyze Particles...", "summarize");
    %clear java
    %MIJ.setTool("multipoint");
    jj
    
end
%inter.batchMode = true;




