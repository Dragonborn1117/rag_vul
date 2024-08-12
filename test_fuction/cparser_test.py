# parser_file 用于处理c语言文件
from pycparser import parse_file
from pycparser import c_ast



class Visitor(c_ast.NodeVisitor):
    def __init__(self):
        self.values = []

    def visit_Constant(self, node):
        # print(f'%s {node.Compound.block_items}')
        self.values.append(node.value)
    
    def visit_FuncDef(self, node):
        print('%s at %s' % (node.decl.name, node.decl.coord))
        
   
def show_func_defs(filename):
    # Note that cpp is used. Provide a path to your own cpp or
    # make sure one exists in PATH.
    ast = parse_file(filename, use_cpp=True,
                     cpp_args=r'-Iutils/fake_libc_include')

    v = Visitor()
    v.generic_visit(ast)

    for Node in ast.ext:
        if Node.__class__.__name__ == 'FuncDef':
            v.visit_FuncDef(Node)
            print(Node)
            # v.visit_Constant(Node)
        

if __name__ == "__main__":

    filename = 'test.cpp'

    show_func_defs(filename)
    