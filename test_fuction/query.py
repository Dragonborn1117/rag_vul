from tree_sitter import Language, Parser

import tree_sitter_c as tsc

# 声明CPP代码解析器
C_LANGUAGE = Language(tsc.language())
parser = Parser(C_LANGUAGE)

cpp_code_snippet = '''
#include <stdio.h>

int cmp(a, b){
	return a > b? 1: 0;
}

int main(){
	int arr[] = {5, 2, 1, 3, 0};
	char s[] = {'h', 'e', 'l', 'l', 'o'};
	int n = 5;
	int i, j, tmp;
	
	for(i = 0; i < n; i++){
		for(j = n - i - 1; j > 0; i--) {
			if(cmp(arr[j - 1], arr[j])){
				tmp = arr[j-1];
				arr[j-1] = arr[j];
				arr[j] = tmp;
			}
		}
	}
	printf("%s\n", s);
	return 0;
}
'''

# 定义query
cpp_query_text = '''
(function_declarator declarator: (identifier)@1 )
(initializer_list) @2
( call_expression ) @3
(assignment_expression  right:(_) @4)
(binary_expression (string_literal)) @b
'''
query = C_LANGUAGE.query(cpp_query_text)

# 获取具体语法树
tree = parser.parse(bytes(cpp_code_snippet, "utf8"))
root_node = tree.root_node

# 获取节点
# capture: list[Node, str]
capture = query.captures(root_node)
for node, alias in capture:
    print(f"{node.text}, {node.type}, @{alias}")

