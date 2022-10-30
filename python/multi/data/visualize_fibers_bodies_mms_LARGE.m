clc; clear all;
set(0,'defaultAxesFontSize',25)
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
set(groot, 'DefaultTextInterpreter', 'latex')


% 0. Initialize
imovie = false; % Flag to make a movie (so it saves images)
movie_reza = false;
nskip = 100; % # of time steps to skip

addpath ./twoCents/run002/
time_system_info_file = 'run002.0.0_time_system_size.txt';
fiber_file{1} = 'run002.0.0_centrosome.1.run5201.0.0_fibers.txt';
fiber_file{2} = 'run002.0.0_centrosome.2.run5201.0.0_fibers.txt';
body_file{1} = 'run002.0.0_centrosome.1.run5201.0.0_clones.txt';
body_file{2} = 'run002.0.0_centrosome.2.run5201.0.0_clones.txt';
force_gen_file = 'run002.0.0.force_generator.txt';
molmotor_head_file = 'run002.0.0_molecular_motors_head.txt';

idraw_nucleus = true;
idraw_mm_heads = false;
NucR = 5.1; % nucleus radius

if imovie
  count = 1;
  mkdir frames
end

% Reza's movie 
if movie_reza
  vidObj = VideoWriter('TwoCentrosomes.avi', 'Uncompressed AVI');
  vidObj.FrameRate = 33;
  open(vidObj);
end

% 1. First load time step sizes, times, # of accepted and rejected tsteps
A = importdata(time_system_info_file,' ',0);
dts = A(:,1); time = A(:,2); naccept = ceil(A(end,3))+1; nreject = ceil(A(end,4));

% 2. EXTRACT SAVED DATA AND RESHAPE THEM FOR PLOTTING
fID_fiber = [];
if ~isempty(fiber_file)
  for i = 1 : length(fiber_file)
    fID_fiber = [fID_fiber; fopen(fiber_file{i})];
  end
else
   xfib = []; yfib = []; zfib = [];
end

fID_body = [];
if ~isempty(body_file)
  for i = 1 : length(body_file)
    fID_body = [fID_body; fopen(body_file{i})];
  end
else
  xc_body = []; yc_body = []; zc_body = [];  
end

fID_forGen = [];
if ~isempty(force_gen_file)
  fID_forGen = fopen(force_gen_file);
end

fID_molmotors = [];
if ~isempty(molmotor_head_file)
  fID_molMot = fopen(molmotor_head_file);    
end


minx = Inf; maxx = -Inf; miny = Inf; maxy = -Inf; minz = Inf; maxz = -Inf;

% Read time steps one by one but not store all of them, skip some so that
% they fit into the memory
if 1
for k = 1 :  naccept
     
  % LOAD FIBER FILES  
  for ifile = 1 : length(fID_fiber)
    C = textscan(fID_fiber(ifile),'%f %f %f %f',1);
    nfibers = ceil(C{1});
    for ifib = 1 : nfibers
        C = textscan(fID_fiber(ifile),'%f %f %f %f',1);
        Nfib = ceil(C{1});
        fib_E_tmp = C{3}; Lfib_tmp = C{4};
        C = textscan(fID_fiber(ifile),'%f %f %f %f', Nfib);
        if rem(k-1,nskip) == 0 % then store data
            fib_E(ifib,(k-1)/nskip+1,ifile) = fib_E_tmp;
            Lfib(ifib,(k-1)/nskip+1,ifile) = Lfib_tmp;
            xfib{ifib,(k-1)/nskip+1,ifile} = C{1};
            yfib{ifib,(k-1)/nskip+1,ifile} = C{2};
            zfib{ifib,(k-1)/nskip+1,ifile} = C{3};
            minx = min(minx,min(xfib{ifib,(k-1)/nskip+1,ifile}));
            maxx = max(maxx,max(xfib{ifib,(k-1)/nskip+1,ifile}));
            miny = min(miny,min(yfib{ifib,(k-1)/nskip+1,ifile}));
            maxy = max(maxy,max(yfib{ifib,(k-1)/nskip+1,ifile}));
            minz = min(minz,min(zfib{ifib,(k-1)/nskip+1,ifile}));
            maxz = max(maxz,max(zfib{ifib,(k-1)/nskip+1,ifile}));
        end
    end
  end
  
  % LOAD BODY FILES
  for ifile = 1 : length(fID_body)
    C = textscan(fID_body(ifile),'%f %f %f %f %f %f %f',1);
    nbodies = C{1};
    for ib = 1 : nbodies
      C = textscan(fID_body(ifile),'%f %f %f %f %f %f %f',2);
      if rem(k-1,nskip) == 0
         body_r(ib,(k-1)/nskip+1,i) = C{1}(1);
         xc_body(ib,(k-1)/nskip+1,i) = C{1}(2);
         yc_body(ib,(k-1)/nskip+1,i) = C{2}(2);
         zc_body(ib,(k-1)/nskip+1,i) = C{3}(2);
         quat_body{i}(1,ib,(k-1)/nskip+1) = C{4}(2);
         quat_body{i}(2,ib,(k-1)/nskip+1) = C{5}(2);
         quat_body{i}(3,ib,(k-1)/nskip+1) = C{6}(2);
         quat_body{i}(4,ib,(k-1)/nskip+1) = C{7}(2);
      end
    end
  end
   
  % LOAD MOLECULAR MOTOR HEADS
  if ~isempty(fID_molMot)
    C = textscan(fID_molMot,'%f %f %f',1);
    n_molMot = C{1};
    molMot_radius = C{2};
    C = textscan(fID_molMot,'%f %f %f',n_molMot);
    if rem(k-1,nskip) == 0
      xc_molMot(:,(k-1)/nskip+1) = C{1};
      yc_molMot(:,(k-1)/nskip+1) = C{2};
      zc_molMot(:,(k-1)/nskip+1) = C{3}; 
    end
  end
    
    
end
save twoCentLargeData body_r xc_body yc_body zc_body quat_body xc_molMot yc_molMot zc_molMot fib_E Lfib xfib yfib zfib minx maxx miny maxy minz maxz
else

load twoCentLargeData
    
    
end
% LOAD FORCE GENERATORS
C = textscan(fID_forGen,'%f %f %f');
nforce = C{1}(1);
forgen_radius = C{2}(1);
xc_forgen = C{1}(2:end);
yc_forgen = C{2}(2:end);
zc_forgen = C{3}(2:end);
    

% 3. PLOT SYSTEM'S EVOLUTION IN TIME
% centrosome geometry

% UPDATE THESE IN FOR LOOP AS WE MOVE IN TIME

[CX, CY, CZ] = sphere(50); % position of points on a sphere

% force-generator geometry
[FGX, FGY, FGZ] = sphere(50); % position of points on a sphere

if idraw_nucleus
  % nucleus geometry
  [NX, NY, NZ] = sphere(100);
  NX = NucR*NX;
  NY = NucR*NY;
  NZ = NucR*NZ;
end

for t = 1 : nskip : naccept
    
    disp([ num2str(t) ' out of ' num2str(naccept) ])

    clf;hold on;
    % Draw nucleus
    if idraw_nucleus
      h = surf(NX, NY, NZ, 'EdgeColor', 'none');
      h.FaceColor = [231/255, 177/255, 33/255];
      h.FaceLighting = 'gouraud';
      h.FaceAlpha = 1;
      h.SpecularStrength = 0.5;    
    end
    
    
    % Draw force generators or molecular motors heads
    if idraw_mm_heads
      for i = 1 : size(xc_molMot,1)
      Xu = FGX*forgen_radius + xc_molMot(i,(t-1)/nskip+1);
      Yu = FGY*forgen_radius + yc_molMot(i,(t-1)/nskip+1);
      Zu = FGZ*forgen_radius + zc_molMot(i,(t-1)/nskip+1);
      h = surf(Xu, Yu, Zu, 'EdgeColor', 'none');
      h.FaceColor = [0.6 0.6 1];
      h.FaceLighting = 'gouraud';
      h.FaceAlpha = 0.5;
      h.AmbientStrength = 0.4;
      h.DiffuseStrength = 0.1;
      h.SpecularStrength = 1;
      h.SpecularExponent = 3;
      h.BackFaceLighting = 'unlit';        
      end

    else
        for i = 1 : size(xc_forgen,1) 
        Xu = FGX*forgen_radius + xc_forgen(i);
        Yu = FGY*forgen_radius + yc_forgen(i);
        Zu = FGZ*forgen_radius + zc_forgen(i);
        h = surf(Xu, Yu, Zu, 'EdgeColor', 'none');
        h.FaceColor = [0.6 0.6 1];
        h.FaceLighting = 'gouraud';
        h.FaceAlpha = 0.5;
        h.AmbientStrength = 0.4;
        h.DiffuseStrength = 0.1;
        h.SpecularStrength = 1;
        h.SpecularExponent = 3;
        h.BackFaceLighting = 'unlit';    
        end
    end
    
    
    % Draw MTs
    for i = 1 : length(fID_fiber)
      for f = 1 : numel(Lfib(:,(t-1)/nskip+1,i))
        xfib_i = xfib{f,(t-1)/nskip+1,i};
        yfib_i = yfib{f,(t-1)/nskip+1,i};
        zfib_i = zfib{f,(t-1)/nskip+1,i};
        [x,y,z] = tubeplot([xfib_i,yfib_i,zfib_i]',0.05, 10);
        h = surf(x, y, z, 'EdgeColor', 'none');
        h.FaceColor = [0.4, 1, 0.4];
        h.FaceLighting = 'gouraud';
        h.FaceAlpha = 0.7;
        h.AmbientStrength = 0.4;
        h.DiffuseStrength = 0.1;
        h.SpecularStrength = 1;
        h.SpecularExponent = 3;
        h.BackFaceLighting = 'unlit';
      end  
    end
    
    % Draw centrosomes
    for i = 1 : length(fID_body)
      for b = 1 : numel(body_r(:,(t-1)/nskip+1,i))
        CX_rad = CX*body_r(b,(t-1)/nskip+1,i);
        CY_rad = CY*body_r(b,(t-1)/nskip+1,i);
        CZ_rad = CZ*body_r(b,(t-1)/nskip+1,i);
        
        % Rotate body based on quaternion
        pvec = quat_body{i}(2:end,b,(t-1)/nskip+1);
        rotMat = 2*(pvec*pvec' + quat_body{i}(1,b,(t-1)/nskip+1)*...
          [0 -pvec(3) pvec(2); pvec(3) 0 -pvec(1); -pvec(2) pvec(1) 0] +...
          (quat_body{i}(1,b,(t-1)/nskip+1)^2-1/2)*eye(3));
        R = cell(51*51,1); R(:) = {rotMat};
        Rsparse = sparse(blkdiag(R{:}));
        Xall = [CX_rad(:)'; CY_rad(:)'; CZ_rad(:)'];
        Xrot = reshape((Rsparse * Xall(:)),3,51*51)';
        
        
        Xbody = reshape(Xrot(:,1),51,51)+xc_body(b,(t-1)/nskip+1,i);
        Ybody = reshape(Xrot(:,2),51,51)+yc_body(b,(t-1)/nskip+1,i);
        Zbody = reshape(Xrot(:,3),51,51)+zc_body(b,(t-1)/nskip+1,i);
        
        h = surf(Xbody, Ybody, Zbody, 'EdgeColor', 'none');
        h.FaceColor = [200/255, 25/255, 33/255];
        h.FaceLighting = 'gouraud';
        h.FaceAlpha = 1;
        h.SpecularStrength = 0.5;
      end
    end
        
        
    % light and view
%     A = 140; B = 20;
    A = 180; B = 10;
    view([A, B]);
    lgt = light('Position',[ 20 20 20], 'Style', 'local');
    lightangle(lgt, 0, 90);
    material dull;
    set(gcf, 'color', 'w');
    pbaspect([1 1 1]);
    set(gca, 'XTick', [], 'YTick', [], 'ZTick', []);
    box off;
    axis off;
    set(gca, 'BoxStyle', 'full');
    %axis equal
    %axis([-6 6 -6 6 -6 6]);   
    axis([-10 10 -10 10 -7 13])
%     xlim([minx-0.1*abs(minx) maxx+0.1*abs(maxx)])
%     ylim([miny-0.1*abs(miny) maxy+0.1*abs(maxy)])
%     zlim([minz-0.1*abs(minz) maxz+0.1*abs(maxz)])  
        
    if imovie
      if movie_reza
        currFrame = getframe(gcf);
        writeVideo(vidObj, currFrame);      
      else
        set(gca,'xtick',[])
        set(gca,'ytick',[])
        set(gca,'ztick',[])
        set(gca,'xcolor','w')
        set(gca,'ycolor','w')
        set(gca,'zcolor','w')
        box on
        set(gca,'visible','off')  
        %titleStr = ['t = ' num2str(time(t)) ];
        %text(0,-1,8,titleStr,'FontSize',14,'FontName','Palatino')
        fileName = ['./frames/image', sprintf('%04d',count),'.png'];
        print(gcf,'-dpng','-r300',fileName)
        count = count + 1;
        figure(1);      
      end
      
    else
      title(['time = ' num2str(time(t))])
      pause(0.1)
    end

    
    
end

