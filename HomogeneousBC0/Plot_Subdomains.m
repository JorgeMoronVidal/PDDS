filepattern = fullfile('Output_*');
Iterations = dir(filepattern);
for l = 1:length(Iterations)
    figure;
    myFolder = Iterations(l).name+"/Subdomains";
    filepattern = fullfile(myFolder, '*.txt');
    files = dir(filepattern);
    labels = [];
    for i=1:length(files)
      if contains(files(i).name, "Sol_")
            label = textscan(files(i).name,"Sol_%s");
            labels = [labels;label{1}{1}];
       end
    end
  [r,c] = size(labels);
  Solution = subplot(1,2,1);
  %Correction = subplot(1,3,2);
  Error = subplot(1,2,2);
  hold(Solution,'on');
  %hold(Correction,'on');
  hold(Error,'on');
  c_axis = [0,0];
  error_cmap = flip(gray);
  MSE = 0;
  ERR = [];
  %MSE_0 = 0;
  %RES = 0;
  %CORR = 0;
  for i=1:r
      label = labels(i,:);
      filename = sprintf("%s/Subdomains/X_%s",Iterations(l).name,labels(i,:));
      x = load(filename,'-ascii');
      len_x = length(x);
      filename = sprintf("%s/Subdomains/Y_%s",Iterations(l).name,labels(i,:));
      y = load(filename,'-ascii');
      len_y = length(y);
      filename = sprintf("%s/Subdomains/Sol_%s",Iterations(l).name,labels(i,:));
      u = load(filename,'-ascii');
      filename = sprintf("%s/Subdomains/Correction_%s",Iterations(l).name,labels(i,:));
      v = load(filename,'-ascii');
      [xx,yy] = meshgrid(x,y);
      uu = reshape(u,len_x,len_y)';
      vv = reshape(v,len_x,len_y)';
      subplot(Solution)
      colormap(Solution,'parula');
      s = surf(xx,yy,uu);
      s.EdgeColor = 'none';
      %subplot(Correction)
      %colormap(Correction,'jet');
      %s = surf(xx,yy,vv);
      %s.EdgeColor = 'none';
      uu_a = ones(size(xx))*0 + sin(pi*xx).*sin(pi*yy);
      %uu_a = Monegros(xx,yy,'u');
      subplot(Error)
      %cyclic_cmap = [gray;flip(gray)];
      colormap(Error, error_cmap);
      err = uu - uu_a;
      if max(abs(c_axis)) < max(max(abs(err)))
          c_axis = [0 max(max(abs(err)))];
          caxis(Error,c_axis);
      end
      s = surf(xx,yy,abs(err));
      s.EdgeColor = 'none';
      ERR = [ERR;err(:)];
      MSE = MSE + (1.0/r)*mean(err(:).^2);
      %MSE_0 = MSE_0 + (1.0/r)*mean(uu_a(:).^2);
  end
   subplot(Solution)
   axis square
   title('Solution')
   set(gca,'FontSize',28)
   xlabel('X')
   ylabel('Y')
   view(2)
   %caxis(Solution,[0 1]);
   colorbar
   %subplot(Correction)
   %axis square
   %title('Solution step 0')
   %set(gca,'FontSize',16)
   %colorbar
   %view(2)
   subplot(Error)
   title('Absolute Error')
   axis square 
   set(gca,'FontSize',28)
   xlabel('X')
   ylabel('Y')
   colorbar
   view(2)
   disp(norm(ERR));
   disp(sqrt(MSE));
end