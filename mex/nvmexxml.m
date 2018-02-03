function  nvmexxml( cuFileName )
if ispc % Windows  
%     CUDA_INC_Location = ['"' getenv('CUDA_PATH')  '\include"'];  
    CUDA_BIN_Location = ['"' getenv('CUDA_PATH')  '\bin"'];
%     CUDA_SAMPLES_Location =['"' getenv('NVCUDASAMPLES8_0_ROOT')  '\common\inc"'];  
    if ( strcmp(computer('arch'),'win32') ==1)  
        CUDA_LIB_Location = ['"' getenv('CUDA_PATH')  '\lib\Win32"'];  
    elseif  ( strcmp(computer('arch'),'win64') ==1)  
        CUDA_LIB_Location = ['"' getenv('CUDA_PATH')  '\lib\x64"']; 
    end  
else % Mac and Linux (assuming gcc is on the path)  
%     CUDA_INC_Location = '/usr/local/cuda/include';  
    CUDA_BIN_Location = ['"' getenv('CUDA_PATH')  '\bin"'];
%     CUDA_SAMPLES_Location = '/usr/local/cuda/samples/common/inc';   
    if ( strcmp(computer('arch'),'win32') ==1)  
        CUDA_LIB_Location = '/usr/local/cuda/lib';  
    elseif  ( strcmp(computer('arch'),'win64') ==1)  
        CUDA_LIB_Location = '/usr/local/cuda/lib64';  
    end  
end  
setenv('CUDA_BIN_PATH', CUDA_BIN_Location);
setenv('CUDA_LIB_PATH', CUDA_LIB_Location);

[~, filename] = fileparts(cuFileName);  
% mexCommandLine = ['mex -g ' filename '.cu'];   %debug in visual studio
mexCommandLine = ['mex ' filename '.cu'];  
warning('off'); 
eval(mexCommandLine);  
warning('on'); 
end

