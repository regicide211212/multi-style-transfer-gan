@echo off
REM 批量图像处理脚本

echo 莫奈风格图像批处理工具
echo ======================
echo.

:menu
echo 请选择处理模式:
echo 1. CycleGAN - 照片转莫奈风格
echo 2. CycleGAN - 莫奈风格转照片
echo 3. 局部风格(简单) - 照片转莫奈风格
echo 4. 局部风格(增强) - 照片转莫奈风格
echo 5. 局部风格(高级) - 照片转莫奈风格
echo 6. 局部风格(简单) - 莫奈风格转照片
echo 7. 局部风格(增强) - 莫奈风格转照片
echo 8. 局部风格(高级) - 莫奈风格转照片
echo 9. 退出
echo.

set /p choice=请输入选项(1-9): 

if "%choice%"=="1" goto cyclegan_photo2monet
if "%choice%"=="2" goto cyclegan_monet2photo
if "%choice%"=="3" goto local_simple_photo2monet
if "%choice%"=="4" goto local_enhanced_photo2monet
if "%choice%"=="5" goto local_advanced_photo2monet
if "%choice%"=="6" goto local_simple_monet2photo
if "%choice%"=="7" goto local_enhanced_monet2photo
if "%choice%"=="8" goto local_advanced_monet2photo
if "%choice%"=="9" goto end

echo 无效选项，请重新选择
goto menu

:cyclegan_photo2monet
echo 执行CycleGAN照片转莫奈风格处理...
python batch_process_images.py --mode cyclegan --direction photo2monet
echo.
echo 处理完成！结果保存在output/batch/cyclegan_photo2monet目录。
goto done

:cyclegan_monet2photo
echo 执行CycleGAN莫奈风格转照片处理...
python batch_process_images.py --mode cyclegan --direction monet2photo
echo.
echo 处理完成！结果保存在output/batch/cyclegan_monet2photo目录。
goto done

:local_simple_photo2monet
echo 执行局部风格(简单)照片转莫奈风格处理...
python batch_process_images.py --mode local_style --local_style_mode simple --direction photo2monet
echo.
echo 处理完成！结果保存在output/batch/local_style_simple_photo2monet目录。
goto done

:local_enhanced_photo2monet
echo 执行局部风格(增强)照片转莫奈风格处理...
python batch_process_images.py --mode local_style --local_style_mode enhanced --direction photo2monet
echo.
echo 处理完成！结果保存在output/batch/local_style_enhanced_photo2monet目录。
goto done

:local_advanced_photo2monet
echo 执行局部风格(高级)照片转莫奈风格处理...
python batch_process_images.py --mode local_style --local_style_mode advanced --direction photo2monet
echo.
echo 处理完成！结果保存在output/batch/local_style_advanced_photo2monet目录。
goto done

:local_simple_monet2photo
echo 执行局部风格(简单)莫奈风格转照片处理...
python batch_process_images.py --mode local_style --local_style_mode simple --direction monet2photo
echo.
echo 处理完成！结果保存在output/batch/local_style_simple_monet2photo目录。
goto done

:local_enhanced_monet2photo
echo 执行局部风格(增强)莫奈风格转照片处理...
python batch_process_images.py --mode local_style --local_style_mode enhanced --direction monet2photo
echo.
echo 处理完成！结果保存在output/batch/local_style_enhanced_monet2photo目录。
goto done

:local_advanced_monet2photo
echo 执行局部风格(高级)莫奈风格转照片处理...
python batch_process_images.py --mode local_style --local_style_mode advanced --direction monet2photo
echo.
echo 处理完成！结果保存在output/batch/local_style_advanced_monet2photo目录。
goto done

:done
pause
goto menu

:end
echo 感谢使用批量处理工具！
pause 