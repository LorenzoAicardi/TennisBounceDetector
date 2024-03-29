%FINAL PROJECT
%DETECT WHEN THE BALL CHANGES DIRECTION

% getting the video
%video = 'tennisvideooo.mp4';
video = 'rafanadal.mp4';

% creating object VideoReader
videoObj = VideoReader(video);


% getting info from the object the number or frames
numFrames = videoObj.NumberOfFrames;

disp(numFrames);

% Parameters
umbralMovimiento = 30;
radioMinimo = 2; 
radioMaximo = 20; 
tamanoMinimoJugador = 50; 
tamanoMaximoJugador = 1000; 


% Inicialize object video
videoPlayer = vision.VideoPlayer;

% Frame processment
for k = 65:120
    % reading consecutive frames
    frameAnterior = read(videoObj, k-1);
    frameActual = read(videoObj, k);
    
    % gray sacle
    frameAnteriorGray = rgb2gray(frameAnterior);
    frameActualGray = rgb2gray(frameActual);
    
    % deference between frames
    diferencia = abs(frameActualGray - frameAnteriorGray);
    

    mascaraMovimiento = diferencia > umbralMovimiento;


    % labeling the regions
    etiquetas = bwlabel(mascaraMovimiento);
    

    % region properties
    propiedades = regionprops(etiquetas, 'Area', 'Centroid', 'MajorAxisLength', 'MinorAxisLength');
    
   % looking for the round regions
    imagenConMovimiento = frameActual;
    for i = 1:length(propiedades)
        % Calculating roundnes and axis
        circularidad = 4 * pi * propiedades(i).Area / (propiedades(i).MajorAxisLength^2);
        relacionEjes = propiedades(i).MinorAxisLength / propiedades(i).MajorAxisLength;
    
        % Establishing ball criteria
        esBola = propiedades(i).MajorAxisLength > radioMinimo && propiedades(i).MajorAxisLength < radioMaximo && circularidad > 0.8 && relacionEjes > 0.7;
        estaEnPosicionBola = propiedades(i).Centroid(2) < size(frameActual, 1) * 0.8; % Ajusta según la ubicación esperada de la bola
        esJugador = propiedades(i).Area > tamanoMinimoJugador && propiedades(i).Area < tamanoMaximoJugador;
    
        % chacking ball criteria
        if esBola && estaEnPosicionBola
            centro = propiedades(i).Centroid;
            radio = propiedades(i).MajorAxisLength / 2;
            imagenConMovimiento = insertShape(imagenConMovimiento, 'Circle', [centro, radio], 'Color', 'green', 'LineWidth', 2);
        elseif esJugador
            % box bounding
            x = propiedades(i).Centroid(1) - propiedades(i).MajorAxisLength / 2;
            y = propiedades(i).Centroid(2) - propiedades(i).MinorAxisLength / 2;
            width = propiedades(i).MajorAxisLength;
            height = propiedades(i).MinorAxisLength;
    
            boundingBox = [x, y, width, height];
    
            % player
            imagenConMovimiento = insertShape(imagenConMovimiento, 'Rectangle', boundingBox, 'Color', 'blue', 'LineWidth', 2);
        end
    end


    
    % movement in red
    % imagenConMovimiento = frameActual;
    % imagenConMovimiento(:, :, 1) = imagenConMovimiento(:, :, 1) + uint8(255 * mascaraMovimiento);
    % imagenConMovimiento(:, :, 2) = imagenConMovimiento(:, :, 2) - uint8(255 * mascaraMovimiento);
    % imagenConMovimiento(:, :, 3) = imagenConMovimiento(:, :, 3) - uint8(255 * mascaraMovimiento);
    
    % Title of the frame
    imagenConMovimiento = insertText(imagenConMovimiento, [10 10], ['Frame ' num2str(k)], 'FontSize', 12, 'TextColor', 'white', 'BoxColor', 'black', 'BoxOpacity', 0.7);
    

    % Visualize result
    step(videoPlayer, imagenConMovimiento);

        
end

% close VideoPlayer
% close(outputVideo);
release(videoPlayer);