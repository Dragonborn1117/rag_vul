#测试

简单记录一下最新的vscode的调试方式

理论上，vscode通过当前目录下的task.json文件构建可执行文件，launch.json文件配置调试任务，但这里使用coderunner插件构建任务更为方便。

在设置中找到插件的ex.map表，CPP需要加上-g选项才能进行gdb调试

目前在launch.json文件中，只需要输入对应语言就可以自动生成模板，因此十分方便