%FINAL PROJECT
%DETECT WHEN THE BALL CHANGES DIRECTION

% getting the video
%video = 'tennisvideooo.mp4';
video = 'final_campeonato.mp4';

% Creating object VideoWriter
outputVideo = VideoWriter('output_video_campeonato_rojo2.avi');
open(outputVideo);

% Creating VideoReader
videoObj = VideoReader(video);


% getting info from the object the number or frames
numFrames = videoObj.NumberOfFrames;

disp(numFrames);

% Parameters 
umbralMovimiento = 30;
radioMinimo = 2; 
radioMaximo = 30; 
tamanoMinimoJugador = 50; 
tamanoMaximoJugador = 1000; 


% Inicialize the object video
videoPlayer = vision.VideoPlayer;

% Processment of the frame
for k = 900:1000
    % reading consecutive frames
    frameAnterior = read(videoObj, k-1);
    frameActual = read(videoObj, k);
    
    % Gray scale
    frameAnteriorGray = rgb2gray(frameAnterior);
    frameActualGray = rgb2gray(frameActual);
    
    % difference between frames
    diferencia = abs(frameActualGray - frameAnteriorGray);
    
    mascaraMovimiento = diferencia > umbralMovimiento;


    % movement label
    etiquetas = bwlabel(mascaraMovimiento);
    

    % calculating region properties
    propiedades = regionprops(etiquetas, 'Area', 'Centroid', 'MajorAxisLength', 'MinorAxisLength');
    
   % looking for cicrles with the size and position that we want
    imagenConMovimiento = frameActual;
    for i = 1:length(propiedades)
        % calculating roundnes and relation with axis
        circularidad = 4 * pi * propiedades(i).Area / (propiedades(i).MajorAxisLength^2);
        relacionEjes = propiedades(i).MinorAxisLength / propiedades(i).MajorAxisLength;
    
        % Establishing criteria
        esBola = propiedades(i).MajorAxisLength > radioMinimo && propiedades(i).MajorAxisLength < radioMaximo && circularidad > 0.8 && relacionEjes > 0.7;
        estaEnPosicionBola = propiedades(i).Centroid(2) < size(frameActual, 1) * 0.8; 
        esJugador = propiedades(i).Area > tamanoMinimoJugador && propiedades(i).Area < tamanoMaximoJugador;
    
        % Checking the criteria
        if esBola && estaEnPosicionBola
            centro = propiedades(i).Centroid;
            radio = propiedades(i).MajorAxisLength / 2;
            imagenConMovimiento = insertShape(imagenConMovimiento, 'Circle', [centro, radio], 'Color', 'green', 'LineWidth', 2);
        elseif esJugador
            % box bounding delimeter
            x = propiedades(i).Centroid(1) - propiedades(i).MajorAxisLength / 2;
            y = propiedades(i).Centroid(2) - propiedades(i).MinorAxisLength / 2;
            width = propiedades(i).MajorAxisLength;
            height = propiedades(i).MinorAxisLength;
    
            boundingBox = [x, y, width, height];
    
            imagenConMovimiento = insertShape(imagenConMovimiento, 'Rectangle', boundingBox, 'Color', 'blue', 'LineWidth', 2);
        end
    end


    
    % % movement in color red
    %  imagenConMovimiento = frameActual;
    %  imagenConMovimiento(:, :, 1) = imagenConMovimiento(:, :, 1) + uint8(255 * mascaraMovimiento);
    %  imagenConMovimiento(:, :, 2) = imagenConMovimiento(:, :, 2) - uint8(255 * mascaraMovimiento);
    %  imagenConMovimiento(:, :, 3) = imagenConMovimiento(:, :, 3) - uint8(255 * mascaraMovimiento);
    
    % title to the frame
    imagenConMovimiento = insertText(imagenConMovimiento, [10 10], ['Frame ' num2str(k)], 'FontSize', 12, 'TextColor', 'white', 'BoxColor', 'black', 'BoxOpacity', 0.7);

    
    % Visualize the result
    step(videoPlayer, imagenConMovimiento);

    % save the frame
    writeVideo(outputVideo, imagenConMovimiento);
        
end

% close the VideoPlayer
close(outputVideo);
release(videoPlayer);