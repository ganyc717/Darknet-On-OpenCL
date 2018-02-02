#ifndef LIST_H
#define LIST_H
#include "darknet.h"

list *make_list();
int list_find(list *l, char *val);

void list_insert(list *, char *);


void free_list_contents(list *l);

#endif
