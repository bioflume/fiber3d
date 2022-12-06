clc; clear all;

addpath ./test4_resume/

n_save = 1;
nskip = 1;

% Colors
fiber_FC = [166/255,217/255, 106/255];
body_FC = [215, 25 ,28]/255;


% Files
time_system_info_file = 'run_time_system_size.txt';

fiber_file = 'run_fibers_fibers.txt';
body_file = 'run_bodies.txt';

body_vel = 'run_body_velocity.txt';

% importdata
A = importdata(time_system_info_file,' ',0);
dts = A(:,1); time = A(:,2); naccept = ceil(A(end,3)); nreject = ceil(A(end,4));
A_fiber = importdata(fiber_file,' ',0);
A_body = importdata(body_file,' ',0);
ntimes = ceil(naccept/n_save); 
A_body_vel = importdata(body_vel, ' ', 0);


% dissect data
offset = 1; offset_body = 1;
nfibers = A_fiber(offset,1);
Nfib = ceil(A_fiber(offset+1,1));
Lfib = zeros(nfibers,1);
xfib = zeros(Nfib,nfibers,ntimes); xfib0 = zeros(Nfib,nfibers);
yfib = zeros(Nfib,nfibers,ntimes); yfib0 = zeros(Nfib,nfibers);
zfib = zeros(Nfib,nfibers,ntimes); zfib0 = zeros(Nfib,nfibers);
for ifib = 1 : nfibers 
  Lfib(ifib,1) = A_fiber(offset+1,3); 
end


nbodies = ceil(A_body(offset_body,1));
body_ra = A_body(offset_body+1,1);
body_rb = A_body(offset_body+1,2);
body_rc = A_body(offset_body+1,3);

xc_body = zeros(nbodies,ntimes); xc0_body = zeros(nbodies,1);
yc_body = zeros(nbodies,ntimes); yc0_body = zeros(nbodies,1);
zc_body = zeros(nbodies,ntimes); zc0_body = zeros(nbodies,1);
quat_body = zeros(4,nbodies,ntimes); quat0_body = zeros(4,nbodies);

% translational velocity
uc_body = A_body_vel(2:2:end,1); vc_body = A_body_vel(2:2:end,2); wc_body = A_body_vel(2:2:end,3);
% angular velocity
ua_body = A_body_vel(2:2:end,4); va_body = A_body_vel(2:2:end,5); wa_body = A_body_vel(2:2:end,6);

% Loop over time steps
% Save the initial configurations
% fibers
for ifib = 1 : nfibers
    
  xfib0(:,ifib) = A_fiber(offset+2:offset+2+Nfib-1,1);
  yfib0(:,ifib) = A_fiber(offset+2:offset+2+Nfib-1,2);
  zfib0(:,ifib) = A_fiber(offset+2:offset+2+Nfib-1,3);

  offset = offset + Nfib + 2;
end
offset = offset + 1;
% Bodies
for ib = 1 : nbodies
  xc0_body(ib) = A_body(offset_body+2,1);
  yc0_body(ib) = A_body(offset_body+2,2);
  zc0_body(ib) = A_body(offset_body+2,3);
  quat0_body(:,ib) = A_body(offset_body+2,4:7);
  offset_body = offset_body + 2;
end
offset_body = offset_body + 1;

for k = 1 : ntimes
  % Fibers 
  for ifib = 1 : nfibers
    
    xfib(:,ifib,k) = A_fiber(offset+2:offset+2+Nfib-1,1);
    yfib(:,ifib,k) = A_fiber(offset+2:offset+2+Nfib-1,2);
    zfib(:,ifib,k) = A_fiber(offset+2:offset+2+Nfib-1,3);

    offset = offset + Nfib + 2;
  end
  offset = offset + 1;
  
  % Bodies
  for ib = 1 : nbodies
    xc_body(ib,k) = A_body(offset_body+2,1);
    yc_body(ib,k) = A_body(offset_body+2,2);
    zc_body(ib,k) = A_body(offset_body+2,3);
    quat_body(:,ib,k) = A_body(offset_body+2,4:7);
    offset_body = offset_body + 2;
  end
  offset_body = offset_body + 1;
end

% PLOTTING
NellPoints = 50;
[CX, CY, CZ] = ellipsoid(0,0,0,body_ra,body_rb,body_rc,NellPoints);
vScale = 100;
count = 1;
for t = 1 : nskip : ntimes
  for ifig = 1 : 1
    figure(ifig);
    clf; hold on;
    for f = 1 : nfibers
      [xf, yf, zf] = tubeplot([xfib(:,f,t), yfib(:,f,t), zfib(:,f,t)]',0.1,10);
      h = surf(xf, yf, zf, 'EdgeColor','none');
      h.FaceColor = fiber_FC;
      h.FaceLighting = 'gouraud';
      h.FaceAlpha = 1;
      h.AmbientStrength = 0.8;
      h.DiffuseStrength = 0.1;
      h.SpecularStrength = 0.5;
      h.SpecularExponent = 3;
    end

    for b = 1 : nbodies
      pvec = quat_body(2:end,b,t);
      rotMat = 2*(pvec*pvec' + quat_body(1,b,t)*...
            [0 -pvec(3) pvec(2); pvec(3) 0 -pvec(1); -pvec(2) pvec(1) 0] +...
            (quat_body(1,b,t)^2-1/2)*eye(3));
       R = cell((NellPoints+1)^2,1); R(:) = {rotMat};
       Rsparse = sparse(blkdiag(R{:}));
       Xall = [CX(:)'; CY(:)'; CZ(:)'];
       Xrot = reshape((Rsparse * Xall(:)),3,(NellPoints+1)^2)';   

       Xbody = reshape(Xrot(:,1),NellPoints+1,NellPoints+1) + xc_body(b,t);
       Ybody = reshape(Xrot(:,2),NellPoints+1,NellPoints+1) + yc_body(b,t);
       Zbody = reshape(Xrot(:,3),NellPoints+1,NellPoints+1) + zc_body(b,t);
        
       h = surf(Xbody, Ybody, Zbody,'EdgeColor','none');
       h.FaceColor = body_FC;
       h.FaceLighting = 'gouraud';
       h.SpecularStrength = 0.5;
       h.FaceAlpha = 0.5;
       if t > 1
         xs = [xc_body(b,t); xc_body(b,t) + 100*uc_body(t)];
         ys = [yc_body(b,t); yc_body(b,t) + 100*vc_body(t)];
         zs = [zc_body(b,t); zc_body(b,t) + 100*wc_body(t)];
         qa = arrow3d(xs, ys, zs, .5, 1, 2, 'm');
       end
    end
  
    % Plot the chebyshev grid for ifig = 1 and cubic grid for ifig = 2
    A = 10; B = 20;
    view([A, B]);
    lgt = light('Position',[ 20 20 20], 'Style', 'local');
    lightangle(lgt, 0, 90);
    material dull;
    set(gcf, 'color', 'w');
    pbaspect([1 1 1]);
    set(gca, 'XTick', [], 'YTick', [], 'ZTick', []);
    set(gca, 'BoxStyle', 'full');
%     set(gca,'visible','off')
    box on
    axis equal
    axis(10*[-1 1 -1 1 -1 1])
    disp(time(t))
    pause
  end % for ifig
  count = count + 1;
end
