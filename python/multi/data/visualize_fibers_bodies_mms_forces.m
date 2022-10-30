clc; clear all;
restoredefaultpath
set(0,'defaultAxesFontSize',25)
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
set(groot, 'DefaultTextInterpreter', 'latex')

n_save = 100; % Should be read from input file
idrawForces = true;
iforcesOnly = false;

% 0. Initialize
imovie = false; % Flag to make a movie (so it saves images)
movie_reza = false;
nskip = 1; % # of time steps to skip
% time_system_info_file = 'run2fibs1body_time_system_size.txt';
% fiber_file{1} = 'run2fibs1body_one_fibers.txt';
% body_file{1} = 'run2fibs1body_one_clones.txt';
force_gen_file = [];

MTs_FC = [166/255,217/255, 106/255; 26/255, 150/255, 65/255];
Cents_FC = [215, 25 ,28]/255;
Nucleus_FC = [255, 255, 193]/255;
MMs_FC = [253, 174, 97] / 255;

if 0
addpath ./twoNuclei/runLongMTs/
time_system_info_file = 'runLongMTs_time_system_size.txt';
fiber_file{1} = 'runLongMTs_centrosome.1.side_fibers.txt';
fiber_file{2} = 'runLongMTs_centrosome.2.side_fibers.txt';
body_file{1} = 'runLongMTs_centrosome.1.side_clones.txt';
body_file{2} = 'runLongMTs_centrosome.2.side_clones.txt';
force_gen_file = 'runLongMTs.force_generator.txt';
molmotor_head_file = 'runLongMTs_molecular_motors_head.txt';
mkdir frames_2nucleiLongMTs
frameFile = './frames_2nucleiLongMTs/image';
NucR = [4.25; 4.25];
NucPos = [7.6; -7.6];
initFrame = 1;
count = 1;
iaxis2nuclei = 1;
iaxis2cents = 0;

fiber_repulsion_force_file{1} = 'runLongMTs_centrosome.1.side_fibers_repulsion_force.txt'; 
fiber_repulsion_force_file{2} = 'runLongMTs_centrosome.2.side_fibers_repulsion_force.txt';
fiber_motor_force_file{1} = 'runLongMTs_centrosome.1.side_fibers_motor_force.txt';
fiber_motor_force_file{2} = 'runLongMTs_centrosome.2.side_fibers_motor_force.txt'; 
body_repulsion_force_file{1} = 'runLongMTs_centrosome.1.side_clones_repulsion_force.txt';
body_repulsion_force_file{2} = 'runLongMTs_centrosome.2.side_clones_repulsion_force.txt';
body_link_force_file{1} = 'runLongMTs_centrosome.1.side_clones_links_force.txt';
body_link_force_file{2} = 'runLongMTs_centrosome.2.side_clones_links_force.txt';
body_link_torque_file{1} = 'runLongMTs_centrosome.1.side_clones_links_torque.txt';
body_link_torque_file{2} = 'runLongMTs_centrosome.2.side_clones_links_torque.txt';

elseif 0
addpath ./twoNuclei/runLongMTs1cent/
time_system_info_file = 'runLongMTs1cent_time_system_size.txt';
fiber_file{1} = 'runLongMTs1cent_centrosome.onlyOne_fibers.txt';
body_file{1} = 'runLongMTs1cent_centrosome.onlyOne_clones.txt';
force_gen_file = 'runLongMTs1cent.force_generator.txt';
molmotor_head_file = 'runLongMTs1cent_molecular_motors_head.txt';
mkdir frames_longMTs1cent
frameFile = './frames_longMTs1cent/image';
NucR = [4.25; 4.25];
NucPos = [7.6; -7.6];
initFrame = 1;
count = 1;
iaxis2nuclei = 1;
iaxis2cents = 0;

fiber_repulsion_force_file{1} = 'runLongMTs1cent_centrosome.onlyOne_fibers_repulsion_force.txt'; 
fiber_motor_force_file{1} = 'runLongMTs1cent_centrosome.onlyOne_fibers_motor_force.txt';
body_repulsion_force_file{1} = 'runLongMTs1cent_centrosome.onlyOne_clones_repulsion_force.txt';
body_link_force_file{1} = 'runLongMTs1cent_centrosome.onlyOne_clones_links_force.txt';
body_link_torque_file{1} = 'runLongMTs1cent_centrosome.onlyOne_clones_links_torque.txt';

elseif 1
addpath ./twoCents/runLongMTs/
time_system_info_file = 'runLongMTs_time_system_size.txt';
fiber_file{1} = 'runLongMTs_centrosome.1.run5201.0.0_fibers.txt';
fiber_file{2} = 'runLongMTs_centrosome.2.run5201.0.0_fibers.txt';
body_file{1} = 'runLongMTs_centrosome.1.run5201.0.0_clones.txt';
body_file{2} = 'runLongMTs_centrosome.2.run5201.0.0_clones.txt';
force_gen_file = 'runLongMTs.force_generator.txt';
molmotor_head_file = 'runLongMTs_molecular_motors_head.txt';
fiber_repulsion_force_file{1} = 'runLongMTs_centrosome.1.run5201.0.0_fibers_repulsion_force.txt'; 
fiber_repulsion_force_file{2} = 'runLongMTs_centrosome.2.run5201.0.0_fibers_repulsion_force.txt';
fiber_motor_force_file{1} = 'runLongMTs_centrosome.1.run5201.0.0_fibers_motor_force.txt';
fiber_motor_force_file{2} = 'runLongMTs_centrosome.2.run5201.0.0_fibers_motor_force.txt'; 
body_repulsion_force_file{1} = 'runLongMTs_centrosome.1.run5201.0.0_clones_repulsion_force.txt';
body_repulsion_force_file{2} = 'runLongMTs_centrosome.2.run5201.0.0_clones_repulsion_force.txt';
body_link_force_file{1} = 'runLongMTs_centrosome.1.run5201.0.0_clones_links_force.txt';
body_link_force_file{2} = 'runLongMTs_centrosome.2.run5201.0.0_clones_links_force.txt';
body_link_torque_file{1} = 'runLongMTs_centrosome.1.run5201.0.0_clones_links_torque.txt';
body_link_torque_file{2} = 'runLongMTs_centrosome.2.run5201.0.0_clones_links_torque.txt';

mkdir frames_1nuclongMTs
frameFile = './frames_1nuclongMTs/image';
NucR = 4.25;
NucPos = 0;
initFrame = 1;
count = 1;
iaxis2nuclei = 0;
iaxis2cents = 1;

end
idraw_mm_heads = ~false;
idraw_nucleus = true;



% Reza's movie 
if movie_reza
  vidObj = VideoWriter('TwoCentrosomes.avi', 'Uncompressed AVI');
  vidObj.FrameRate = 33;
  open(vidObj);
end

if 1
% 1. First load time step sizes, times, # of accepted and rejected tsteps
A = importdata(time_system_info_file,' ',0);
dts = A(:,1); time = A(:,2); naccept = ceil(A(end,3))+1; nreject = ceil(A(end,4));


% 2. EXTRACT SAVED DATA AND RESHAPE THEM FOR PLOTTING
A_fiber = [];
if ~isempty(fiber_file)
  for i = 1 : length(fiber_file)
    A_fiber{i} = importdata(fiber_file{i},' ',0);
  end
else
  A_fiber = [];  
end

A_fiber_repulsion = [];
if ~isempty(fiber_repulsion_force_file)
  for i = 1 : length(fiber_repulsion_force_file)
    A_fiber_repulsion{i} = importdata(fiber_repulsion_force_file{i},' ',0);
  end
else
  A_fiber_repulsion = [];  
end

A_fiber_motor = [];
if ~isempty(fiber_motor_force_file)
  for i = 1 : length(fiber_motor_force_file)
    A_fiber_motor{i} = importdata(fiber_motor_force_file{i},' ',0);
  end
else
  A_fiber_motor = [];  
end

A_body_repulsion = [];
if ~isempty(body_repulsion_force_file)
  for i = 1 : length(body_repulsion_force_file)
    A_body_repulsion{i} = importdata(body_repulsion_force_file{i},' ',0);
  end
else
  A_body_repulsion = [];  
end

A_body_link_force = [];
if ~isempty(body_link_force_file)
  for i = 1 : length(body_link_force_file)
    A_body_link_force{i} = importdata(body_link_force_file{i},' ',0);
  end
else
  A_body_link_force = [];  
end

A_body_link_torque = [];
if ~isempty(body_link_torque_file)
  for i = 1 : length(body_link_torque_file)
    A_body_link_torque{i} = importdata(body_link_torque_file{i},' ',0);
  end
else
  A_body_link_torque = [];  
end

if ~isempty(body_file)
  for i = 1 : length(body_file)
    A_body{i} = importdata(body_file{i},' ',0);
  end
else
  A_body = [];
end

A_forGen = [];
if ~isempty(force_gen_file)
  A_forGen = importdata(force_gen_file,' ',0);
end

A_molMot = [];
if ~isempty(molmotor_head_file)
  A_molMot = importdata(molmotor_head_file,' ',0);
end

if idrawForces
offset = ones(length(A_fiber_repulsion),1);
offset_body = ones(length(A_body_repulsion),1);
for k = 1 : (naccept-1)/n_save
  for i = 1 : length(A_fiber_repulsion)
    nfibers = A_fiber_repulsion{i}(offset(i),1);
    for ifib = 1 : nfibers
      Nfib = ceil(A_fiber_repulsion{i}(offset(i)+1,1));
      fiber_repul_x{ifib,k,i} = A_fiber_repulsion{i}(offset(i)+2:offset(i)+2+Nfib-1,1);
      fiber_repul_y{ifib,k,i} = A_fiber_repulsion{i}(offset(i)+2:offset(i)+2+Nfib-1,2);
      fiber_repul_z{ifib,k,i} = A_fiber_repulsion{i}(offset(i)+2:offset(i)+2+Nfib-1,3);
    
      fiber_motor_x{ifib,k,i} = A_fiber_motor{i}(offset(i)+2:offset(i)+2+Nfib-1,1);
      fiber_motor_y{ifib,k,i} = A_fiber_motor{i}(offset(i)+2:offset(i)+2+Nfib-1,2);
      fiber_motor_z{ifib,k,i} = A_fiber_motor{i}(offset(i)+2:offset(i)+2+Nfib-1,3);
      
      offset(i) = offset(i)+Nfib+1;
    end
    offset(i) = offset(i) + 1;
  end
  
  for i = 1 : length(A_body_repulsion)
    nbodies = ceil(A_body{i}(offset_body(i),1));
    for ib = 1 : nbodies
      body_repulsion_x(ib,k,i) = A_body_repulsion{i}(offset_body(i)+2-k,1);
      body_repulsion_y(ib,k,i) = A_body_repulsion{i}(offset_body(i)+2-k,2);
      body_repulsion_z(ib,k,i) = A_body_repulsion{i}(offset_body(i)+2-k,3);
      
      body_link_force_x(ib,k,i) = A_body_link_force{i}(offset_body(i)+2-k,1);
      body_link_force_y(ib,k,i) = A_body_link_force{i}(offset_body(i)+2-k,2);
      body_link_force_z(ib,k,i) = A_body_link_force{i}(offset_body(i)+2-k,3);
      
      body_link_torque_x(ib,k,i) = A_body_link_torque{i}(offset_body(i)+2-k,1);
      body_link_torque_y(ib,k,i) = A_body_link_torque{i}(offset_body(i)+2-k,2);
      body_link_torque_z(ib,k,i) = A_body_link_torque{i}(offset_body(i)+2-k,3);
      offset_body(i) = offset_body(i)+2;
    end
    offset_body(i) = offset_body(i) + 1;
  end
end
end

minx = Inf; maxx = -Inf; miny = Inf; maxy = -Inf; minz = Inf; maxz = -Inf;
offset = ones(length(A_fiber),1); offset_body = ones(length(A_body),1);
offset_for = 1;
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
  
  if ~isempty(molmotor_head_file)
    nforce = A_molMot(offset_for,1);
    molMot_radius(k) = A_molMot(offset_for,2);
    xc_molMot(:,k) = A_molMot(offset_for+1:offset_for+nforce,1);
    yc_molMot(:,k) = A_molMot(offset_for+1:offset_for+nforce,2);
    zc_molMot(:,k) = A_molMot(offset_for+1:offset_for+nforce,3);
    offset_for = offset_for + nforce + 1;
  else
    xc_molmot = [];
  end

  

end % k = 1 : naccept
end
% 3. PLOT SYSTEM'S EVOLUTION IN TIME
% centrosome geometry

% LOAD FORCE GENERATORS
if ~isempty(A_forGen)
nforce = A_forGen(1,1);
forgen_radius = A_forGen(1,2)*0.5;
xc_forgen = A_forGen(2:end,1);
yc_forgen = A_forGen(2:end,2);
zc_forgen = A_forGen(2:end,3);
end
    
% UPDATE THESE IN FOR LOOP AS WE MOVE IN TIME

[CX, CY, CZ] = sphere(50); % position of points on a sphere

% force-generator geometry
[FGX, FGY, FGZ] = sphere(50); % position of points on a sphere

if idraw_nucleus
  % nucleus geometry
  [NX, NY, NZ] = sphere(100);
end

for t = initFrame : (naccept-1)/n_save + 1
    
    disp([ num2str(t) ' out of ' num2str((naccept-1)/n_save) ])

    clf;hold on;
    % Draw nucleus
    if idraw_nucleus
      for i = 1 : numel(NucR)
        NX2 = NucR(i)*NX;
        NY2 = NucR(i)*NY;
        NZ2 = NucR(i)*NZ + NucPos(i);  
        h = surf(NX2, NY2, NZ2, 'EdgeColor', 'none');
        %h.FaceColor = [231/255, 177/255, 33/255];
        h.FaceColor = Nucleus_FC;
        h.FaceLighting = 'gouraud';
        h.FaceAlpha = 1;
        h.SpecularStrength = 0.5;    
      end
    end
    
    % Draw force generators or molecular motors heads
    if idraw_mm_heads
      for i = 1 : size(xc_molMot,1)
      Xu = FGX*forgen_radius + xc_molMot(i,t);
      Yu = FGY*forgen_radius + yc_molMot(i,t);
      Zu = FGZ*forgen_radius + zc_molMot(i,t);
      if ~idrawForces
      h = surf(Xu, Yu, Zu, 'EdgeColor', 'none');
      %h.FaceColor = [0.6 0.6 1];
      h.FaceColor = MMs_FC;
      h.FaceLighting = 'gouraud';
      h.FaceAlpha = 0.5;
      h.AmbientStrength = 0.4;
      h.DiffuseStrength = 0.1;
      h.SpecularStrength = 1;
      h.SpecularExponent = 3;
      h.BackFaceLighting = 'unlit';
      else
      plot3(xc_molMot(i,t),yc_molMot(i,t),zc_molMot(i,t),'o','markersize',12,'markerfacecolor',MMs_FC,'markeredgecolor',MMs_FC)
      end
      
      end

    else
        for i = 1 : size(xc_forgen,1) 
        Xu = FGX*forgen_radius + xc_forgen(i);
        Yu = FGY*forgen_radius + yc_forgen(i);
        Zu = FGZ*forgen_radius + zc_forgen(i);
        h = surf(Xu, Yu, Zu, 'EdgeColor', 'none');
        %h.FaceColor = [0.6 0.6 1];
        h.FaceColor = MMs_FC;
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
    for i = 1 : length(A_fiber)
      for f = 1 : numel(Lfib(:,t,i))
        xfib_i = xfib{f,t,i};
        yfib_i = yfib{f,t,i};
        zfib_i = zfib{f,t,i};
        if ~idrawForces
        [x,y,z] = tubeplot([xfib_i,yfib_i,zfib_i]',0.05, 10);
        h = surf(x, y, z, 'EdgeColor', 'none');
        %h.FaceColor = [0.4, 1, 0.4];
        h.FaceColor = MTs_FC(i,:);
        h.FaceLighting = 'gouraud';
        h.FaceAlpha = 0.7;
        h.AmbientStrength = 0.4;
        h.DiffuseStrength = 0.1;
        h.SpecularStrength = 1;
        h.SpecularExponent = 3;
        h.BackFaceLighting = 'unlit';
        end
        
        if ~idrawForces
        [x,y,z] = tubeplot([xfib_i,yfib_i,zfib_i]',0.05, 10);
        h = surf(x, y, z, 'EdgeColor', 'none');
        %h.FaceColor = [0.4, 1, 0.4];
        h.FaceColor = MTs_FC(i,:);
        h.FaceLighting = 'gouraud';
        h.FaceAlpha = 0.7;
        h.AmbientStrength = 0.4;
        h.DiffuseStrength = 0.1;
        h.SpecularStrength = 1;
        h.SpecularExponent = 3;
        h.BackFaceLighting = 'unlit';
        elseif t > 1
        plot3(xfib_i,yfib_i,zfib_i,'linewidth',2,'Color',MTs_FC(i,:))
        fib_rep_x = fiber_repul_x{f,t-1,i};
        fib_rep_y = fiber_repul_y{f,t-1,i};
        fib_rep_z = fiber_repul_z{f,t-1,i};
          
        fib_mot_x = fiber_motor_x{f,t-1,i};
        fib_mot_y = fiber_motor_y{f,t-1,i};
        fib_mot_z = fiber_motor_z{f,t-1,i};
        
        quiver3(xfib_i,yfib_i,zfib_i,fib_rep_x,fib_rep_y,fib_rep_z,'r','linewidth',2)
        quiver3(xfib_i,yfib_i,zfib_i,fib_mot_x,fib_mot_y,fib_mot_z,'k','linewidth',2)
        end
        
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
        %h.FaceColor = [200/255, 25/255, 33/255];
        h.FaceColor = Cents_FC;
        h.FaceLighting = 'gouraud';
        h.FaceAlpha = 1;
        h.SpecularStrength = 0.5;
        
        if idrawForces && t > 1
        h.FaceAlpha = 0.6;
        bdy_rep_x = body_repulsion_x(b,t-1,i);
        bdy_rep_y = body_repulsion_x(b,t-1,i);
        bdy_rep_z = body_repulsion_z(b,t-1,i);
          
        bdy_link_x = body_link_force_x(b,t-1,i);
        bdy_link_y = body_link_force_y(b,t-1,i);
        bdy_link_z = body_link_force_z(b,t-1,i);
        
        quiver3(xc_body(b,t,i),yc_body(b,t,i),zc_body(b,t,i),bdy_rep_x,bdy_rep_y,bdy_rep_z,10,'b','linewidth',2)
        quiver3(xc_body(b,t,i),yc_body(b,t,i),zc_body(b,t,i),bdy_link_x,bdy_link_y,bdy_link_z,10,'k','linewidth',2)       
        
        disp('Body forces:')
        [bdy_rep_x bdy_link_x]
        [bdy_rep_y bdy_link_y]
        [bdy_rep_z bdy_link_z]
        end
        
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
    if iaxis2nuclei
      %axis equal
      axis([-14 14 -14 14 -14 14])
    elseif iaxis2cents
      axis([-10 10 -10 10 -7 13])
    else
      axis equal
      axis([-5 5 -5 5 -5 5])
    end

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
        fileName = [frameFile, sprintf('%04d',count),'.png'];
        print(gcf,'-dpng','-r300',fileName)
        count = count + 1;
        figure(1);      
      end
      
    else
      title(['time = ' num2str(time(t))])
      pause(0.1)
    end

    
    
end

