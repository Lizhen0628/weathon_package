# -*- coding: utf-8 -*-
# @Time    : 2022/10/6 14:32
# @Author  : LiZhen
# @FileName: states_machine.py
# @github  : https://github.com/Lizhen0628
# @Description:
from copy import deepcopy


class StatesMachineException(Exception):
    pass


# (START, END, FAIL, WAIT_TAIL) = list(range(4))
# (START, END, FAIL, WAIT_TAIL) = 0，1，2，3
# # conditions
# (TAIL, ERROR, MATCHED_SWITCH, UNMATCHED_SWITCH, CONNECTOR) = list(range(5))

class StatesMachine(object):
    # states: start,end,fail,wait_tail
    START, END, FAIL, WAIT_TAIL = 0, 1, 2, 3
    # condition:tail,error,matched_switch,unmatched_switch,connector
    TAIL, ERROR, MATCHED_SWITCH, UNMATCHED_SWITCH, CONNECTOR = 0, 2, 3, 4, 5

    def __init__(self):
        self.state = self.START
        self.final = ""
        self.len = 0
        self.pool = ""

    def clone(self, pool):
        new = deepcopy(self)
        new.state = self.WAIT_TAIL
        new.pool = pool
        return new

    def feed(self, char, map):
        node = map[self.pool + char]

        if node.have_child:
            if node.is_tail:
                if node.is_original:
                    cond = self.UNMATCHED_SWITCH
                else:
                    cond = self.MATCHED_SWITCH
            else:
                cond = self.CONNECTOR
        else:
            if node.is_tail:
                cond = self.TAIL
            else:
                cond = self.ERROR

        new = None
        if cond == self.ERROR:
            self.state = self.FAIL
        elif cond == self.TAIL:
            if self.state == self.WAIT_TAIL and node.is_original_long_word():
                self.state = self.FAIL
            else:
                self.final += node.to_word
                self.len += 1
                self.pool = ""
                self.state = self.END
        elif self.state == self.START or self.state == self.WAIT_TAIL:
            if cond == self.MATCHED_SWITCH:
                new = self.clone(node.from_word)
                self.final += node.to_word
                self.len += 1
                self.state = self.END
                self.pool = ""
            elif cond == self.UNMATCHED_SWITCH or cond == self.CONNECTOR:
                if self.state == self.START:
                    new = self.clone(node.from_word)
                    self.final += node.to_word
                    self.len += 1
                    self.state = self.END
                else:
                    if node.is_follow(self.pool):
                        self.state = self.FAIL
                    else:
                        self.pool = node.from_word
        elif self.state == self.END:
            # END is a new START
            self.state = self.START
            new = self.feed(char, map)
        elif self.state == self.FAIL:
            raise StatesMachineException('Translate States Machine '
                                         'have error with input data %s' % node)
        return new

    def __len__(self):
        return self.len + 1

    def __str__(self):
        return '<StatesMachine %s, pool: "%s", state: %s, final: %s>' % (
            id(self), self.pool, self.state, self.final)

    __repr__ = __str__
