import sys,os
import ast
from operator import itemgetter
from inspect import currentframe, getframeinfo


class ResultsFile(object):
    """
    This class plays with the results file.
    When constructing, pass the path of the results file. For example, "../wt2g/all_baseline_results/okapi_0.05"
    """
    def __init__(self, path):
        self.path = os.path.abspath(path)
        if not os.path.exists(self.path):
            frameinfo = getframeinfo(currentframe())
            print frameinfo.filename, frameinfo.lineno
            print '[ResultsFile Constructor]:Please provide a valid results file path'
            exit(1)


    def go_through_results_file(self, callback, callback_paras):
        """
        Go through the result file line by line.
        A callback function processes with each line.
        callback_paras are the parameters pass to callback function
        """

        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = line.split()
                    _qid = row[0]
                    doc_id = row[2]
                    score = ast.literal_eval(row[4])
                    if callback:
                        callback_paras.append([_qid, doc_id, score])
                        #print callback_paras
                        callback(*callback_paras)
                        callback_paras.pop(-1)


    def test_callback(*paras):
        for ele in paras:
            print ele


    def get_all_results_callback(self, cut, format, results, row_of_line):
        """
        @Input : 
            qid - qid
            cut - as of get_all_results
            format - as of get_all_results
            results - for the return
            row_of_line - one line in result file after split into [qid, docid, score]
        """
        qid = row_of_line[0]
        doc_id = row_of_line[1]
        score = row_of_line[2]
        if qid not in results:
            if format == 'tuple' or format == 'list': 
                results[qid] = []
            elif format == 'dict':
                results[qid] = {}

        if cut < 0 or (cut >=0 and len(results[qid]) < cut):
            if format == 'tuple':
                results[qid].append((doc_id, score))
            elif format == 'list':
                results[qid].append([doc_id, score])
            elif format == 'dict':
                results[qid][doc_id] = score


    def get_all_results(self, cut=-1, format='tuple'):
        """
        get the results of all queries

        @Input: 
            cut : how many documents will be retrieved per query, -1 mean no limits
            format (string) : The return format: as "tuple" [(qid, score),...], 
                "dict" {qid:score,...}, "list" [[qid, score],...]

        @Return: a dict {qid:[(docid, score), ...], ...}
        """
        results = {}
        self.go_through_results_file(self.get_all_results_callback, [cut, format, results])
        return results


    def get_all_results_as_list(self, cut=-1):
        """
        get the results of all queries as a single list

        @Input: 
            cut : how many documents will be retrieved per query, -1 mean no limits

        @Return: a list [(qid, docid, score), ...]
        """
        results = []
        all_r = self.get_all_results(cut, 'list')
        for qid in all_r:
            for ele in all_r[qid]:
                results.append( (qid, ele[0], str(ele[1])) )
        return results

    def get_results_of_some_queries(self, qids, cut=-1, format='tuple'):
        """
        get the results of a single query

        @Input: 
            qids (list) : a list contains the qid that to be returned
            cut (int) : how many documents will be retrieved, -1 mean no limits
            format (string) : The return format: as "tuple" [(qid, score),...], 
                "dict" {qid:score,...}, "list" [[qid, score],...]

        @Return: As parameter format indicates (See Above)
        """

        all_results = self.get_all_results(cut, format)
        return {k: all_results.get(k, None) for k in qids}


if __name__ == '__main__':
    j = ResultsFile('../../wt2g/all_baseline_results/okapi_0.05')
    #j.go_through_judgment_file(j.test_callback, ['a', 'b', 'c'])
    print j.get_results_of_some_queries(['403', '405'], 5)
    raw_input()
    print j.get_results_of_some_queries(['403'], 5, 'dict')


