%FINAL PROJECT
%DETECT WHEN THE BALL CHANGES DIRECTION

% getting the video
video = 'nadal1.mp4';

% Creating an object VideoWriter
%outputVideo = VideoWriter('output_video.avi');
%open(outputVideo);

% Creating an object VideoReader
videoObj = VideoReader(video);


% getting info from the object the number or frames
numFrames = videoObj.NumberOfFrames;

disp(numFrames);

% Parameters 
umbralMovimiento = 30;
radioMinimo = 2; 
radioMaximo = 4; 
tamanoMinimoJugador = 800; 
tamanoMaximoJugador = 2000; 


% Inicialize video object
videoPlayer = vision.VideoPlayer;

% frame processing
for k = 2:200 
    % reading consecutive frames
    frameAnterior = read(videoObj, k-1);
    frameActual = read(videoObj, k);
    
    % Changing to gray scale
    frameAnteriorGray = rgb2gray(frameAnterior);
    frameActualGray = rgb2gray(frameActual);
    
    % difference between frames
    diferencia = abs(frameActualGray - frameAnteriorGray);
    
    
    mascaraMovimiento = diferencia > umbralMovimiento;


    % labels 
    etiquetas = bwlabel(mascaraMovimiento);
    

    % Calculating regions
    propiedades = regionprops(etiquetas, 'Area', 'Centroid', 'MajorAxisLength', 'MinorAxisLength');
    
   % defining the circular movement
    imagenConMovimiento = frameActual;
    for i = 1:length(propiedades)
        % calculating the circle
        circularidad = 4 * pi * propiedades(i).Area / (propiedades(i).MajorAxisLength^2);
        relacionEjes = propiedades(i).MinorAxisLength / propiedades(i).MajorAxisLength;
    
        % establishing the criteria
        esBola = propiedades(i).MajorAxisLength > radioMinimo && propiedades(i).MajorAxisLength < radioMaximo && circularidad > 0.8 && relacionEjes > 0.7;
        estaEnPosicionBola = propiedades(i).Centroid(2) < size(frameActual, 1) * 0.8; % Ajusta según la ubicación esperada de la bola
        esJugador = propiedades(i).Area > tamanoMinimoJugador && propiedades(i).Area < tamanoMaximoJugador;
    
        % looking if it fits the conditions
        if esBola && estaEnPosicionBola
            centro = propiedades(i).Centroid;
            radio = propiedades(i).MajorAxisLength / 2;
            imagenConMovimiento = insertShape(imagenConMovimiento, 'Circle', [centro, radio], 'Color', 'green', 'LineWidth', 2);
        elseif esJugador
            % delimiting the region
            x = propiedades(i).Centroid(1) - propiedades(i).MajorAxisLength / 2;
            y = propiedades(i).Centroid(2) - propiedades(i).MinorAxisLength / 2;
            width = propiedades(i).MajorAxisLength;
            height = propiedades(i).MinorAxisLength;
    
            boundingBox = [x, y, width, height];
    
            % player
            imagenConMovimiento = insertShape(imagenConMovimiento, 'Rectangle', boundingBox, 'Color', 'blue', 'LineWidth', 2);
        end
    end


    
    % movement detection
    % imagenConMovimiento = frameActual;
    % imagenConMovimiento(:, :, 1) = imagenConMovimiento(:, :, 1) + uint8(255 * mascaraMovimiento);
    % imagenConMovimiento(:, :, 2) = imagenConMovimiento(:, :, 2) - uint8(255 * mascaraMovimiento);
    % imagenConMovimiento(:, :, 3) = imagenConMovimiento(:, :, 3) - uint8(255 * mascaraMovimiento);
    
    % adding title to the frame
    imagenConMovimiento = insertText(imagenConMovimiento, [10 10], ['Frame ' num2str(k)], 'FontSize', 12, 'TextColor', 'white', 'BoxColor', 'black', 'BoxOpacity', 0.7);
    imwrite(imagenConMovimiento, sprintf('frame_%d.png', k));

    % Visualize the result
    step(videoPlayer, imagenConMovimiento);

    % Saving the actual frame in the output video
    %writeVideo(outputVideo, imagenConMovimiento);
        
end

% Closing VideoPlayer
%close(outputVideo);
release(videoPlayer);