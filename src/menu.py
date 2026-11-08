"""Interactive shortcuts for the existing CLI; heavy tasks run in child processes."""
import os
import subprocess
import sys
import webbrowser

from .utils import ROOT, load_config, path, read_json


def choose_run(cfg):
    runs = []
    for file in path(cfg['results_dir']).glob('*/run.json'):
        try:
            state = read_json(file)
            if state.get('status') == 'complete':
                runs.append((file.parent, file.stat().st_mtime))
        except (OSError, ValueError):
            continue
    runs.sort(key=lambda row: row[1], reverse=True)
    if not runs:
        print('还没有完成的实验，请先选择“新建检索实验”。')
        return None
    print('\n请选择实验（按时间从新到旧）：')
    for number, (run, _) in enumerate(runs, 1):
        print(f'  {number}. {run.name}')
    while True:
        choice = input('输入编号，回车选择第 1 个，0 返回：').strip() or '1'
        if choice == '0':
            return None
        if choice.isdecimal() and 1 <= int(choice) <= len(runs):
            return runs[int(choice) - 1][0]
        print('请输入列表中的编号。')


def execute(config_file, *arguments):
    result = subprocess.run(
        [sys.executable, '-u', str(ROOT / 'main.py'), '--config', str(config_file), *arguments],
        cwd=ROOT,
    )
    if result.returncode:
        print(f'操作未完成（退出码 {result.returncode}），请查看上面的提示。')
    return result.returncode == 0


def open_report(run):
    report = run / 'report' / 'index.html'
    if not report.is_file():
        raise ValueError('报告不存在，请先通过 report 命令导出。')
    print(f'打开报告：{report}')
    if os.name == 'nt':
        os.startfile(str(report))
    else:
        webbrowser.open(report.resolve().as_uri())


def menu(config_file):
    cfg = load_config(config_file)
    print('\nRAG 项目 · 中文操作菜单')
    print('查看已有报告无需重新运行实验。')
    while True:
        print('\n  1. 查看已有报告（直接打开浏览器）'
              '\n  2. 检查运行环境'
              '\n  3. 新建检索实验（四种方法）'
              '\n  0. 退出')
        try:
            choice = input('输入数字：').strip()
            if choice == '0':
                return
            if choice == '1':
                run = choose_run(cfg)
                if run is not None:
                    open_report(run)
            elif choice == '2':
                execute(config_file, 'doctor')
            elif choice == '3':
                print('将使用当前已准备的数据，新建实验并保留旧结果。')
                execute(config_file, 'run')
            else:
                print('请输入 0～3。')
        except EOFError:
            return
        except KeyboardInterrupt:
            print('\n已返回菜单，已保存的结果保留。')
        except (OSError, ValueError) as exc:
            print(f'操作未完成：{exc}')
