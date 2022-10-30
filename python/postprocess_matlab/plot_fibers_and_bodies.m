clear; clc;
isaveimage = false;
if isaveimage
  count = 1;
  mkdir frames
end

% function visualize(runName,iplot,skip,isavefig,irunInfoReady)

nfibers  = 2;    % # of fibers
nbodies  = 1;    % # of bodies
Nfiber   = 16;   % # of points per fiber
Nbody    = 64;   % # of points per rigid body
body_dia = 2;    % diameter of rigid body
dt       = 5e-3; % time step size
nsteps   = 5001;  % # of time step size
skip     = 20;    % # of time steps to skip in plotting

% if there are more than 1 type of fiber, then make this a cell
fib_file = 'run2fibers16pts_wReparam_deg6_one.fibers.txt';
bod_file = 'run2fibers16pts_wReparam_deg6_one.clones.txt';

% load fiber info
delimiterIn = ' ';
headerlinesIn = 0;
A = importdata(fib_file,delimiterIn,headerlinesIn);
minx = Inf; maxx = -Inf;
miny = Inf; maxy = -Inf;
minz = Inf; maxz = -Inf;

offset = 1;
for k = 1 : nsteps
  for f = 1 : nfibers
    npoints = A(offset,1);
    npoints_store(f,k) = npoints;
    if npoints == 0 || offset + npoints >= numel(A(:,1))
      nsteps = k - 1;
      break;
    end
    
    xfib{f,k} = A(offset+1:offset+npoints,1);
    yfib{f,k} = A(offset+1:offset+npoints,2);
    zfib{f,k} = A(offset+1:offset+npoints,3);
    
    minx = min(minx,min(xfib{f,k}));
    maxx = max(maxx,max(xfib{f,k}));
    miny = min(miny,min(yfib{f,k}));
    maxy = max(maxy,max(yfib{f,k}));
    minz = min(minz,min(zfib{f,k}));
    maxz = max(maxz,max(zfib{f,k}));
    
    if isempty(xfib{f,k})
      nsteps = k-1;
      break;
    end
    len = compute_length([xfib{f,k};yfib{f,k};zfib{f,k}]);
    len_store(f,k) = len;
    
    offset = offset + npoints+1;
    if offset > numel(A(:,1)) 
      nsteps = k;
    end
  end
end

minx = 1.1*min(minx,-body_dia/2);
miny = 1.1*min(miny,-body_dia/2);
maxx = 1.1*max(maxx,body_dia/2);
maxy = 1.1*max(maxy,body_dia/2);

if 0
xfib = reshape(A(1:Nfiber*nfibers*nsteps,1),Nfiber,nfibers,nsteps);
yfib = reshape(A(1:Nfiber*nfibers*nsteps,2),Nfiber,nfibers,nsteps);
zfib = reshape(A(1:Nfiber*nfibers*nsteps,3),Nfiber,nfibers,nsteps);
tension = reshape(A(1:Nfiber*nfibers*nsteps,4),Nfiber,nfibers,nsteps);
end

% load body info
A = importdata(bod_file,delimiterIn,headerlinesIn);
cx_bod = A(:,1); cy_bod = A(:,2); cz_bod = A(:,3); % centers
quat_bod = A(:,4:7);

% draw a body
[xs,ys,zs] = sphere(Nbody);
xs = xs * body_dia/2; ys = ys * body_dia/2; zs = zs * body_dia/2;

% draw walls
xwall = linspace(minx,maxx,100);
ywall = linspace(miny,maxy,100);

[Xwall,Ywall] = meshgrid(xwall,ywall);
Zwall = ones(size(Xwall));

for tt = 1 : skip : nsteps
  figure(1);clf;
  X = zeros(Nfiber+1,Nfiber+2,nfibers);
  Y = zeros(Nfiber+1,Nfiber+2,nfibers);
  Z = zeros(Nfiber+1,Nfiber+2,nfibers);
  for ifib = 1 : nfibers
    %len = compute_length([xfib(:,ifib,tt);yfib(:,ifib,tt);zfib(:,ifib,tt)]);
    
    %[X(:,:,ifib),Y(:,:,ifib),Z(:,:,ifib)] = tubeplot([xfib(:,ifib,tt) yfib(:,ifib,tt) zfib(:,ifib,tt)]',len*0.01,Nfiber);
    %[X(:,:,ifib),Y(:,:,ifib),Z(:,:,ifib)] = tubeplot([xfib{ifib,tt} yfib{ifib,tt} zfib{ifib,tt}]',len*0.01,Nfiber);
  end
  
  % plot bodies
  xsNew = zeros(Nbody,1); ysNew = zeros(Nbody,1); zsNew = zeros(Nbody,1);
  % rotate xs based on quaternion
  pvec = quat_bod(tt,2:end)';
  rotMat = 2*(pvec*pvec' + quat_bod(tt,1)*...
      [0 -pvec(3) pvec(2); pvec(3) 0 -pvec(1); -pvec(2) pvec(1) 0] + (quat_bod(tt,1)^2-1/2)*eye(3));
  for isp = 1 : Nbody + 1
    for jsp = 1 : Nbody + 1
      newpos = rotMat* [xs(isp,jsp);ys(isp,jsp);zs(isp,jsp)];
      xsNew(isp,jsp) = newpos(1); ysNew(isp,jsp) = newpos(2); zsNew(isp,jsp) = newpos(3);
    end
  end
  h1 = surf(xsNew+cx_bod(tt),ysNew+cy_bod(tt),zsNew+cz_bod(tt),'facecolor',[0.635 0.078 0.184],'edgecolor','none','facealpha',.6);
  hold on
  %plot3(xsNew(1:10,1:10)+cx_bod(tt),ysNew(1:10,1:10)+cy_bod(tt),zsNew(1:10,1:10)+cz_bod(tt),...
  %    'o','markerfacecolor',[1 1 1],'markersize',2)
  plot3(xsNew+cx_bod(tt),ysNew+cy_bod(tt),zsNew+cz_bod(tt),...
      'o','markersize',1)
  % plot fibers
  for ifib = 1 : nfibers
  plot3(xfib{ifib,tt}, yfib{ifib,tt}, zfib{ifib,tt},'Color',[0 0.447 0.741], 'linewidth',3)
  %h2 = surf(X(:,:,ifib),Y(:,:,ifib),Z(:,:,ifib),'facecolor',[0 0.447 0.741],'edgecolor','none');
  end
  axis equal
  view(140,20)
  
  h3 = surf(Xwall,Ywall,Zwall*0,'facecolor',[.5 .5 .5], 'edgecolor','none','facealpha',0.3);
  h4 = surf(Xwall,Ywall,Zwall*6,'facecolor',[.5 .5 .5], 'edgecolor','none','facealpha',0.3);
  
  xlim([minx maxx])
  ylim([miny maxy])
  zlim([-0.05 6.05])
  
%   xlim([-2 2])
%   ylim([-2 20])
%   zlim([0, 8])
  
  
  title(['t = ' num2str((tt-1)*dt) ', fiber length = ' num2str(max(len_store(:,tt)))])
  
  
  if isaveimage
    set(gca,'xtick',[])
    set(gca,'ytick',[])
    set(gca,'ztick',[])
    set(gca,'xcolor','w')
    set(gca,'ycolor','w')
    set(gca,'zcolor','w')
    box on
    set(gca,'visible','off')  
    titleStr = ['t = ' num2str((tt-1)*dt) ', fiber length = '  num2str(max(len_store(:,tt)))];
    text(0,-12,12,titleStr,'FontSize',14,'FontName','Palatino')
    fileName = ['./frames/image', sprintf('%04d',count),'.png'];
    print(gcf,'-dpng','-r300',fileName)
    count = count + 1;
    figure(1);
  else
    pause(0.1)
  end
  
  
end


function length = compute_length(x)
N = numel(x)/3;
[D,alpha] = cheb(N-1);

alpha = flipud(alpha);
D = flipud(flipud(D')');

xa = D*x(1:N);
ya = D*x(N+1:2*N);
za = D*x(2*N+1:3*N);

length = trapz(alpha,sqrt(xa.^2+ya.^2+za.^2));

end

function  [D,x]= cheb(N) 
if N==0,   D=0; x=1; return,  end
x=cos(pi*(0:N)/N)';   
c=[2;ones(N-1,1);2].*(-1).^(0:N)';      
X=repmat(x,1,N+1);
dX=X-X';
D=(c*(1./c)')./(dX+eye(N+1));%off-diagonal entries 
D=D-diag(sum(D'));%diagonal entries 

end