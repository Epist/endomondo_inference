import copy
class Sequence_Frame(object):
    def __init__(self, predictions_by_workout, targets_by_workout, inputs_by_workout, data_epoch):
        
        self.isSequence = {'altitude':True, 'gender':False, 'heart_rate':True, 'id':False, 'latitude':True, 'longitude':True,
                             'speed':True, 'sport':False, 'timestamp':True, 'url':False, 'userId':False, 'time_elapsed': True, 
                             'distance':True, 'new_workout':True, 'derived_speed':True}
        
        self.values = [] # A list of attributes each of which is a list of workouts where each workout is a len 500 list of timesteps in the workout
        self.attributes = []
        
        #[self.values.append(x) for x in predictions_by_workout]
        self.values.append(predictions_by_workout)
        self.attributes.append("predictions")
        
        #[self.values.append(x) for x in targets_by_workout]
        self.values.append(targets_by_workout)
        self.attributes.append("targets")
        
        #For each input attribute, add the attribute list to the values and attributes lists
        inputAtts = [x for x in data_epoch.endoFeatures if x != data_epoch.targetAtt]
        for i, att in enumerate(inputAtts):
            current_att_inputs_by_workout = inputs_by_workout[att]
            if self.isSequence[att]:
                #[self.values.append(x) for x in current_att_inputs_by_workout]
                self.values.append(current_att_inputs_by_workout)
            else:
                #[self.values.append([x[0]]) for x in current_att_inputs_by_workout] #Use only one copy of attribute variables
                #[self.values.append(x) for x in current_att_inputs_by_workout]
                self.values.append([[x[0]] for x in current_att_inputs_by_workout])
                #self.values.append(current_att_inputs_by_workout)
            self.attributes.append(att)
            
        self.num_rows = len(self.values[0])
        print("Num rows: " + str(self.num_rows))
        
    def get_row(self, row_num):
        row = [x[row_num] for x in self.values]
        return row
    
    def get_col(self, col_num):
        col = self.values[col_num]
        return col
    
    def get_col_by_att(self, att):
        col_num = self.attributes.index(att)
        return self.get_col(col_num)
    
    def slice_rows_by_att_vals(self, atts, vals, colvars=None):
        """Takes a list of attributes and a list of values of the same length and returns the rows that
        match the values given for each of the atts given (an intersection)
        Returns all the columns if colvars is None, otherwise it returns the columns in the colvars list
        """
        #Start with a full row_list
        #for each attribute, go through the row_list, removing the entries that do not contain the value at that attribute
        
        row_list = range(self.num_rows) #Ideally, this should be a linked list, since it is only traversed and node deleted,
                                        #never randomly accessed...
        
        for i, att in enumerate(atts):
            tempRowList = copy.deepcopy(row_list)
            for row_num in row_list:
                current_row = self.get_row(row_num)
                if current_row[self.attributes.index(att)][0] != vals[i]:
                    tempRowList.remove(row_num)
            row_list = tempRowList
        
        #Now get all the remaining rows and return them as a list of rows
        sliced_rows = []
        for row_num in row_list:
            if colvars is None:
                sliced_rows.append(self.get_row(row_num))
            else:
                current_row = self.get_row(row_num)
                sliced_rows.append([current_row[self.attributes.index(att)] for att in colvars])
                
        return sliced_rows
