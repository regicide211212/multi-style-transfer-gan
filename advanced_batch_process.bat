@echo off
REM 高级批量图像处理脚本

echo 莫奈风格图像高级批处理工具
echo ==========================
echo.

:input_dir
set input_dir=test_images
set /p custom_input=请输入图像目录[默认: test_images]: 
if not "%custom_input%"=="" set input_dir=%custom_input%

:output_dir
set output_dir=output/batch
set /p custom_output=请输入输出目录[默认: output/batch]: 
if not "%custom_output%"=="" set output_dir=%custom_output%

:mode_select
echo.
echo 请选择处理模式:
echo 1. CycleGAN
echo 2. 局部风格
echo.
set /p mode_choice=请输入选项(1-2): 

if "%mode_choice%"=="1" (
    set mode=cyclegan
    goto direction_select
)
if "%mode_choice%"=="2" (
    set mode=local_style
    goto local_style_mode
)

echo 无效选项，请重新选择
goto mode_select

:local_style_mode
echo.
echo 请选择局部风格模式:
echo 1. 简单模式 (simple)
echo 2. 增强模式 (enhanced)
echo 3. 高级模式 (advanced)
echo.
set /p local_mode_choice=请输入选项(1-3): 

if "%local_mode_choice%"=="1" set local_style_mode=simple
if "%local_mode_choice%"=="2" set local_style_mode=enhanced
if "%local_mode_choice%"=="3" set local_style_mode=advanced

if not defined local_style_mode (
    echo 无效选项，请重新选择
    goto local_style_mode
)

:direction_select
echo.
echo 请选择转换方向:
echo 1. 照片转莫奈风格 (Photo to Monet)
echo 2. 莫奈风格转照片 (Monet to Photo)
echo.
set /p direction_choice=请输入选项(1-2): 

if "%direction_choice%"=="1" set direction=photo2monet
if "%direction_choice%"=="2" set direction=monet2photo

if not defined direction (
    echo 无效选项，请重新选择
    goto direction_select
)

REM 如果选择的是局部风格模式，继续设置其他参数
if "%mode%"=="local_style" (
    goto strength_select
) else (
    goto confirm
)

:strength_select
echo.
set strength=0.8
set /p custom_strength=请输入风格强度(0.0-1.0)[默认: 0.8]: 
if not "%custom_strength%"=="" set strength=%custom_strength%

:detail_select
echo.
set detail=0.7
set /p custom_detail=请输入细节保留水平(0.0-1.0)[默认: 0.7]: 
if not "%custom_detail%"=="" set detail=%custom_detail%

:enhance_colors
echo.
echo 是否增强颜色:
echo 1. 是
echo 2. 否
echo.
set enhance_colors=--enhance_colors
set /p enhance_choice=请输入选项(1-2)[默认: 是]: 

if "%enhance_choice%"=="2" set enhance_colors=--no_enhance_colors

:smooth_transitions
echo.
echo 是否平滑过渡:
echo 1. 是
echo 2. 否
echo.
set smooth=--smooth
set /p smooth_choice=请输入选项(1-2)[默认: 是]: 

if "%smooth_choice%"=="2" set smooth=--no_smooth

:confirm
echo.
echo === 处理配置 ===
echo 输入目录: %input_dir%
echo 输出目录: %output_dir%
echo 处理模式: %mode%
if "%mode%"=="local_style" (
    echo 局部风格模式: %local_style_mode%
    echo 风格强度: %strength%
    echo 细节保留: %detail%
    if "%enhance_colors%"=="--enhance_colors" (
        echo 增强颜色: 是
    ) else (
        echo 增强颜色: 否
    )
    if "%smooth%"=="--smooth" (
        echo 平滑过渡: 是
    ) else (
        echo 平滑过渡: 否
    )
    set specific_output_dir=%output_dir%/local_style_%local_style_mode%_%direction%
) else (
    set specific_output_dir=%output_dir%/cyclegan_%direction%
)
echo 转换方向: %direction%
echo 输出将保存在: %specific_output_dir%
echo.
echo 请确认以上设置:
echo 1. 确认并开始处理
echo 2. 重新设置
echo.
set /p confirm_choice=请输入选项(1-2): 

if "%confirm_choice%"=="1" goto process
if "%confirm_choice%"=="2" goto input_dir

echo 无效选项，请重新选择
goto confirm

:process
echo.
echo 开始处理图像...

if "%mode%"=="cyclegan" (
    python batch_process_images.py --input_dir %input_dir% --output_dir %output_dir% --mode %mode% --direction %direction%
) else (
    python batch_process_images.py --input_dir %input_dir% --output_dir %output_dir% --mode %mode% --local_style_mode %local_style_mode% --direction %direction% --strength %strength% --detail %detail% %enhance_colors% %smooth%
)

echo.
echo 处理完成！结果保存在 %specific_output_dir% 目录。
echo.

:ask_again
echo 是否继续处理其他图像?
echo 1. 是
echo 2. 否
echo.
set /p continue_choice=请输入选项(1-2): 

if "%continue_choice%"=="1" goto input_dir
if "%continue_choice%"=="2" goto end

echo 无效选项，请重新选择
goto ask_again

:end
echo 感谢使用高级批量处理工具！
pause 