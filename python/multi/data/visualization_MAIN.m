clc; clear all;
restoredefaultpath
set(0,'defaultAxesFontSize',25)
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
set(groot, 'DefaultTextInterpreter', 'latex')
addpath ./simulation_data_info_mfiles/

% 0. Initialize
imovie = ~false; % Flag to make a movie (so it saves images)
reload_data = true;
nskip = 1; % # of time steps to skip
n_save = 100; % Should be read from input file


% Load the run files' names
%run('twoCents_run200FG')
%run('twoCents_runLongMTsMovingMM')
%run('oneCent_runFixed.m')
%run('twoCents_runLongMTsMovingMM.m')
%run('twoCents_runLongMTsNew.m')
%run('twoCents_runTEST_STAT.m')
%run('twoNuclei_run1cent.m')
%run('twoNuclei_runFIXED.m')
%run('twoNuclei_runLongMTs1cent.m')
%run('twoNuclei_runLongMTs.m')
%run('twoNuclei_runLongMTsMovingMM_1cent.m')
%run('twoNuclei_runLongMTsMovingMM_1centNew.m')
%run('twoNuclei_runLongMTsMovingMM.m')
%run('twoNuclei_runLongSides.m')
%run('twoNuclei_runMoving_Equil.m')
%run('oneCent_runMoving.m')
%run('oneCentNew_runMovingResume.m')
% run('oneCent_fixNucleus_Cortex.m')
% run('twoNuclei_twoCents.m')
%run('twoNuclei_twoCents_moving.m')
% run('oneCent_oneNuc.m')
%run('twoFarNuclei_twoCents_movingCoarse.m')
% run('twoFarNuclei_twoCents_FewLongFixed.m')
% run('twoFarNuclei_twoCents_FewLongMoving.m')
%run('one500MT.m')


% Colors setup
MTs_FC = [166/255,217/255, 106/255; 26/255, 150/255, 65/255];
Cents_FC = [215, 25 ,28]/255;
Nucleus_FC = [255, 255, 193]/255;
MMs_FC = [253, 174, 97] / 255;


if reload_data
% 1. First load time step sizes, times, # of accepted and rejected tsteps
A = importdata(time_system_info_file,' ',0);
dts = A(:,1); time = A(:,2); naccept = ceil(A(end,3))+1; nreject = ceil(A(end,4));
cortex_radius = [];
if numel(A(1,:)) > 4
  cortex_radius = A(1,5);
end

% 2. EXTRACT SAVED DATA AND RESHAPE THEM FOR PLOTTING
% -------------------------------------------------------------------------
A_fiber = [];
if ~isempty(fiber_file)
  for i = 1 : length(fiber_file)
    A_fiber{i} = importdata(fiber_file{i},' ',0);
  end
else
  A_fiber = [];  
end

if ~isempty(body_file)
  for i = 1 : length(body_file)
    A_body{i} = importdata(body_file{i},' ',0);
  end
else
  A_body = [];
end

if ~isempty(nucleus_file)
  for i = 1 : length(nucleus_file)
    A_nucleus{i} = importdata(nucleus_file{i},' ',0);
  end
else
  A_nucleus = [];
end

A_molMot_base = [];
if ~isempty(molmotor_base_file)
  for i = 1 : length(molmotor_base_file)
    A_molMot_base{i} = importdata(molmotor_base_file{i},' ',0);
  end
end

A_molMot_head = [];
if ~isempty(molmotor_head_file)
  for i = 1 : length(molmotor_head_file)
    A_molMot_head{i} = importdata(molmotor_head_file{i},' ',0);
  end
end

A_attached_ends = [];
if ~isempty(attached_ends_file)
  for i = 1 : length(attached_ends_file)
    A_attached_ends{i} = importdata(attached_ends_file{i},' ',0);
  end
end

A_forGen = [];
if ~isempty(forgen_file)
  A_forGen = importdata(forgen_file, ' ', 0);    
end


% -------------------------------------------------------------------------

minx = Inf; maxx = -Inf; miny = Inf; maxy = -Inf; minz = Inf; maxz = -Inf;
offset = ones(length(A_fiber),1); offset_body = ones(length(A_body),1);
offset_nucleus = ones(length(A_nucleus),1);
offset_mm = ones(length(A_molMot_head),1);
for k = 1 : (naccept-1)/n_save 
  if ~isempty(A_fiber)
    for i = 1 : length(A_fiber)
      nfibers = A_fiber{i}(offset(i),1); % # of fibers in this step
      for ifib = 1 : nfibers
        Nfib = ceil(A_fiber{i}(offset(i)+1,1));
        fib_E(ifib,k,i) = A_fiber{i}(offset(i)+1,3); 
        Lfib(ifib,k,i) = A_fiber{i}(offset(i)+1,4);
        xfib{ifib,k,i} = A_fiber{i}(offset(i)+2:offset(i)+2+Nfib-1,1);
        yfib{ifib,k,i} = A_fiber{i}(offset(i)+2:offset(i)+2+Nfib-1,2);
        zfib{ifib,k,i} = A_fiber{i}(offset(i)+2:offset(i)+2+Nfib-1,3);
        
        minx = min(minx,min(xfib{ifib,k,i}));
        maxx = max(maxx,max(xfib{ifib,k,i}));
        miny = min(miny,min(yfib{ifib,k,i}));
        maxy = max(maxy,max(yfib{ifib,k,i}));
        minz = min(minz,min(zfib{ifib,k,i}));
        maxz = max(maxz,max(zfib{ifib,k,i}));
    
        offset(i) = offset(i)+Nfib+1;
      end % ifib = 1 : nfibers
      offset(i) = offset(i) + 1;
    end
  else
    xfib = []; yfib = []; zfib = [];
  end % ~isempty(A_fiber)
  
  if ~isempty(A_body)
    for i = 1 : length(A_body)
      nbodies = ceil(A_body{i}(offset_body(i),1));
      for ib = 1 : nbodies
        body_r(ib,k,i) = A_body{i}(offset_body(i)+1,2); 
        xc_body(ib,k,i) = A_body{i}(offset_body(i)+2,1);
        yc_body(ib,k,i) = A_body{i}(offset_body(i)+2,2);
        zc_body(ib,k,i) = A_body{i}(offset_body(i)+2,3);
        quat_body{i}(:,ib,k) = A_body{i}(offset_body(i)+2,4:7);
        offset_body(i) = offset_body(i)+2;
      end
      offset_body(i) = offset_body(i) + 1;
    end
  else
    xc_body = []; yc_body = []; zc_body = [];  
  end
  
  if ~isempty(A_nucleus)
    for i = 1 : length(A_nucleus)
      nnucleus = ceil(A_nucleus{i}(offset_nucleus(i),1));
      for ib = 1 : nnucleus
        nucleus_r(ib,k,i) = A_nucleus{i}(offset_nucleus(i)+1,2); 
        xc_nucleus(ib,k,i) = A_nucleus{i}(offset_nucleus(i)+2,1);
        yc_nucleus(ib,k,i) = A_nucleus{i}(offset_nucleus(i)+2,2);
        zc_nucleus(ib,k,i) = A_nucleus{i}(offset_nucleus(i)+2,3);
        quat_nucleus{i}(:,ib,k) = A_nucleus{i}(offset_nucleus(i)+2,4:7);
        offset_nucleus(i) = offset_nucleus(i)+2;
      end
      offset_nucleus(i) = offset_nucleus(i) + 1;
    end
  else
    xc_nucleus = []; yc_nucleus = []; zc_nucleus = [];  
  end
  
  if ~isempty(molmotor_head_file)
    for i = 1 : length(molmotor_head_file)
      nforce = A_molMot_head{i}(offset_mm(i),1);
      molMot_radius(k,i) = A_molMot_head{i}(offset_mm(i),2);
      xc_molMot_head(:,k,i) = A_molMot_head{i}(offset_mm(i)+1:offset_mm(i)+nforce,1);
      yc_molMot_head(:,k,i) = A_molMot_head{i}(offset_mm(i)+1:offset_mm(i)+nforce,2);
      zc_molMot_head(:,k,i) = A_molMot_head{i}(offset_mm(i)+1:offset_mm(i)+nforce,3);
      if ~isempty(molmotor_base_file)
      xc_molMot_base(:,k,i) = A_molMot_base{i}(offset_mm(i)+1:offset_mm(i)+nforce,1);
      yc_molMot_base(:,k,i) = A_molMot_base{i}(offset_mm(i)+1:offset_mm(i)+nforce,2);
      zc_molMot_base(:,k,i) = A_molMot_base{i}(offset_mm(i)+1:offset_mm(i)+nforce,3);
      else
      xc_molMot_base = [];
      end
      
      if ~isempty(A_attached_ends)
      attached_end_base(:,k,i) = A_attached_ends{i}(offset_mm(i)+1:offset_mm(i)+nforce,1);
      attached_end_head(:,k,i) = A_attached_ends{i}(offset_mm(i)+1:offset_mm(i)+nforce,2);
      else
      attached_end_base = []; attached_end_head = [];
      end
      offset_mm(i) = offset_mm(i) + nforce + 1;
    end
  else
    xc_molMot_head = []; xc_molMot_base = [];
  end

end % k = 1 : naccept
end

% 3. PLOT SYSTEM'S EVOLUTION IN TIME

% LOAD FORCE GENERATORS
forgen_radius = 0.25;
if ~isempty(A_forGen)
nforce = A_forGen(1,1);
xc_forgen = A_forGen(2:end,1);
yc_forgen = A_forGen(2:end,2);
zc_forgen = A_forGen(2:end,3);
end
    
% UPDATE THESE IN FOR LOOP AS WE MOVE IN TIME

[CX, CY, CZ] = sphere(50); % position of points on a sphere

% force-generator geometry
[FGX, FGY, FGZ] = sphere(50); % position of points on a sphere

% nucleus geometry
[NX, NY, NZ] = sphere(50);

% Colors setup
MTs_FC = [166/255,217/255, 106/255; 26/255, 150/255, 65/255];
Cents_FC = [215, 25 ,28]/255;
Nucleus_FC = [255, 255, 193]/255;
MMs_FC = [253, 174, 97] / 255;
Cort_FC = [239,243,255] / 255;

% Draw cortex
if ~isempty(cortex_radius)
  cortx = NX * cortex_radius;
  corty = NY * cortex_radius;
  cortz = NZ * cortex_radius;
end

for t = initFrame :nskip: (naccept-1)/n_save
    
    disp([ num2str(t) ' out of ' num2str((naccept-1)/n_save) ])

    clf;hold on;
    % Draw cortex
    if ~isempty(cortex_radius)
      h = surf(cortx, corty, cortz, 'EdgeColor','None');
      h.FaceColor = Cort_FC;
      h.FaceLighting = 'gouraud';
      h.FaceAlpha = 0.2;
      h.SpecularStrength = 0.3;
    end
      
    % Draw nucleus
    if isempty(A_nucleus)
    for i = 1 : numel(NucR)
      NX2 = NucR(i)*NX;
      NY2 = NucR(i)*NY;
      NZ2 = NucR(i)*NZ + NucPos(i);  
      h = surf(NX2, NY2, NZ2, 'EdgeColor', 'none');     
      h.FaceColor = Nucleus_FC;
      h.FaceLighting = 'gouraud';
      h.FaceAlpha = 1;
      h.SpecularStrength = 0.5;    
    end    
    else
    for i = 1 : length(A_nucleus)
      for b = 1 : numel(nucleus_r(:,t,i))
        CX_rad = NX*nucleus_r(b,t,i);
        CY_rad = NY*nucleus_r(b,t,i);
        CZ_rad = NZ*nucleus_r(b,t,i);
        
        % Rotate body based on quaternion
        pvec = quat_nucleus{i}(2:end,b,t);
        rotMat = 2*(pvec*pvec' + quat_nucleus{i}(1,b,t)*...
          [0 -pvec(3) pvec(2); pvec(3) 0 -pvec(1); -pvec(2) pvec(1) 0] +...
          (quat_nucleus{i}(1,b,t)^2-1/2)*eye(3));
        R = cell(51*51,1); R(:) = {rotMat};
        Rsparse = sparse(blkdiag(R{:}));
        Xall = [CX_rad(:)'; CY_rad(:)'; CZ_rad(:)'];
        Xrot = reshape((Rsparse * Xall(:)),3,51*51)'; 
        
        Xnuc = reshape(Xrot(:,1),51,51)+xc_nucleus(b,t,i);
        Ynuc = reshape(Xrot(:,2),51,51)+yc_nucleus(b,t,i);
        Znuc = reshape(Xrot(:,3),51,51)+zc_nucleus(b,t,i);
        
        h = surf(Xnuc, Ynuc, Znuc, 'EdgeColor', 'none');
        h.FaceColor = Nucleus_FC;
        h.FaceLighting = 'gouraud';
        h.FaceAlpha = 1;
        h.SpecularStrength = 0.5;
      end
    end
    end
    
    % Draw MMs
    for imm = 1 : size(xc_molMot_base,3)
      for i = 1 : size(xc_molMot_head,1)
       if ~isempty(attached_end_head)
         active = false;
         if attached_end_head(i,t,imm) > -1
           active = true;
         end
       else
       if t > 1
         dist_mot = sqrt((xc_molMot_head(i,t,imm)-xc_molMot_head(i,t-1,imm))^2+...
            (yc_molMot_head(i,t,imm)-yc_molMot_head(i,t-1,imm))^2+...
            (zc_molMot_head(i,t,imm)-zc_molMot_head(i,t-1,imm))^2);
      
         active = true;
         if dist_mot <= 1e-10; active = false; end
       else
         active = false;
       end
       end
      
       if active
         motorx = xc_molMot_head(i,t,imm);
         motory = yc_molMot_head(i,t,imm);
         motorz = zc_molMot_head(i,t,imm);
       else
         if ~isempty(xc_molMot_base)
         motorx = xc_molMot_head(i,t,imm);
         motory = yc_molMot_head(i,t,imm);
         motorz = zc_molMot_head(i,t,imm);  
         else
         motorx = xc_forgen(i);
         motory = yc_forgen(i);
         motorz = zc_forgen(i);    
         end
       end
       if ~isempty(xc_molMot_base)
         base_motor_x = xc_molMot_base(i,t,imm);
         base_motor_y = yc_molMot_base(i,t,imm);
         base_motor_z = zc_molMot_base(i,t,imm);
       else
         base_motor_x = xc_forgen(i);
         base_motor_y = yc_forgen(i);
         base_motor_z = zc_forgen(i);
       end
       Xu = FGX*forgen_radius + motorx;
       Yu = FGY*forgen_radius + motory;
       Zu = FGZ*forgen_radius + motorz;
       line_x = linspace(base_motor_x,motorx,10)';
       line_y = linspace(base_motor_y,motory,10)';
       line_z = linspace(base_motor_z,motorz,10)';
       plot3(line_x,line_y,line_z,'Color',[.5 .5 .5], 'linewidth',2)
       h = surf(Xu, Yu, Zu, 'EdgeColor', 'none');
       h.FaceColor = MMs_FC;
       h.FaceLighting = 'gouraud';
       if active
         h.FaceAlpha = 0.95;
       else
         h.FaceAlpha = 0.25;
       end
       h.AmbientStrength = 0.4;
       h.DiffuseStrength = 0.1;
       h.SpecularStrength = 1;
       h.SpecularExponent = 3;
       h.BackFaceLighting = 'unlit';        
      end
    end
    
    % Draw MTs
    for i = 1 : length(A_fiber)
      for f = 1 : numel(Lfib(:,t,i))
        xfib_i = xfib{f,t,i};
        yfib_i = yfib{f,t,i};
        zfib_i = zfib{f,t,i};
        [x,y,z] = tubeplot([xfib_i,yfib_i,zfib_i]',0.05, 10);
        h = surf(x, y, z, 'EdgeColor', 'none');
        h.FaceColor = MTs_FC(i,:);
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
    for i = 1 : length(A_body)
      for b = 1 : numel(body_r(:,t,i))
        CX_rad = CX*body_r(b,t,i);
        CY_rad = CY*body_r(b,t,i);
        CZ_rad = CZ*body_r(b,t,i);
        
        % Rotate body based on quaternion
        pvec = quat_body{i}(2:end,b,t);
        rotMat = 2*(pvec*pvec' + quat_body{i}(1,b,t)*...
          [0 -pvec(3) pvec(2); pvec(3) 0 -pvec(1); -pvec(2) pvec(1) 0] +...
          (quat_body{i}(1,b,t)^2-1/2)*eye(3));
        R = cell(51*51,1); R(:) = {rotMat};
        Rsparse = sparse(blkdiag(R{:}));
        Xall = [CX_rad(:)'; CY_rad(:)'; CZ_rad(:)'];
        Xrot = reshape((Rsparse * Xall(:)),3,51*51)';
        
        
        Xbody = reshape(Xrot(:,1),51,51)+xc_body(b,t,i);
        Ybody = reshape(Xrot(:,2),51,51)+yc_body(b,t,i);
        Zbody = reshape(Xrot(:,3),51,51)+zc_body(b,t,i);
        
        h = surf(Xbody, Ybody, Zbody, 'EdgeColor', 'none');
        h.FaceColor = Cents_FC;
        h.FaceLighting = 'gouraud';
        h.FaceAlpha = 1;
        h.SpecularStrength = 0.5;
      end
    end
        
        
    % light and view
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
    if iaxis2nuclei
      axis([-14 14 -14 14 -14 14])
    elseif iaxis2cents
      axis([-10 10 -10 10 -7 13])
    else
      %axis equal
      axis([-21 21 -21 21 -30 12])
    end
        
    if imovie
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
      fileName = [frameFile, sprintf('%04d',count),'.png'];
      print(gcf,'-dpng','-r300',fileName)
      count = count + 1;
      figure(1);      
    else
      title(['time = ' num2str(time(t))])
      pause(0.1)
    end

end

