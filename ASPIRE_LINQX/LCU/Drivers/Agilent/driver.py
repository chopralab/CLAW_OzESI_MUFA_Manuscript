import pyautogui


class PopulateRows(self):
    __ROW_1_NAME_POSITION = (0,0)
    __ROW_1_SAMPLE_POSITION = (0,0)
    __ROW_1_METHOD_POSITION = (0,0)
    __ROW_1_DATAFILE_POSITION = (0,0)

    __ROW_2_NAME_POSITION = (0,0)
    __ROW_2_SAMPLE_POSITION = (0,0)
    __ROW_2_METHOD_POSITION = (0,0)
    __ROW_2_DATAFILE_POSITION = (0,0)

    __ROW_3_NAME_POSITION = (0,0)
    __ROW_3_SAMPLE_POSITION = (0,0)
    __ROW_3_METHOD_POSITION = (0,0)
    __ROW_3_DATAFILE_POSITION = (0,0)

    def __populate_row_1(name, sample, method, datafile):
        # Populate the name field on MassHunter Worklist
        pyautogui.moveTo(PopulateRows.__ROW_1_NAME_POSITION[0],PopulateRows.__ROW_1_NAME_POSITION[1])
        pyautogui.click()
        pyautogui.write(name)

        pyautogui.moveTo(PopulateRows.__ROW_1_SAMPLE_POSITION[0],PopulateRows.__ROW_1_SAMPLE_POSITION[1])
        pyautogui.click()
        pyautogui.write(sample)

        pyautogui.moveTo(PopulateRows.__ROW_1_METHOD_POSITION[0],PopulateRows.__ROW_1_METHOD_POSITION[1])
        pyautogui.click()
        pyautogui.write(method)

        pyautogui.moveTo(PopulateRows.__ROW_1_DATAFILE_POSITION[0],PopulateRows.__ROW_1_DATAFILE_POSITION[1])
        pyautogui.click()
        pyautogui.write(datafile)

    def __populate_row_2(name, sample, method, datafile):
        # Populate the name field on MassHunter Worklist
        pyautogui.moveTo(PopulateRows.__ROW_2_NAME_POSITION[0],PopulateRows.__ROW_2_NAME_POSITION[1])
        pyautogui.click()
        pyautogui.write(name)

        pyautogui.moveTo(PopulateRows.__ROW_2_SAMPLE_POSITION[0],PopulateRows.__ROW_2_SAMPLE_POSITION[1])
        pyautogui.click()
        pyautogui.write(sample)

        pyautogui.moveTo(PopulateRows.__ROW_2_METHOD_POSITION[0],PopulateRows.__ROW_2_METHOD_POSITION[1])
        pyautogui.click()
        pyautogui.write(method)

        pyautogui.moveTo(PopulateRows.__ROW_2_DATAFILE_POSITION[0],PopulateRows.__ROW_2_DATAFILE_POSITION[1])
        pyautogui.click()
        pyautogui.write(datafile)

    def __populate_row_3(name, sample, method, datafile):
        # Populate the name field on MassHunter Worklist
        pyautogui.moveTo(PopulateRows.__ROW_3_NAME_POSITION[0],PopulateRows.__ROW_3_NAME_POSITION[1])
        pyautogui.click()
        pyautogui.write(name)

        pyautogui.moveTo(PopulateRows.__ROW_3_SAMPLE_POSITION[0],PopulateRows.__ROW_3_SAMPLE_POSITION[1])
        pyautogui.click()
        pyautogui.write(sample)

        pyautogui.moveTo(PopulateRows.__ROW_3_METHOD_POSITION[0],PopulateRows.__ROW_3_METHOD_POSITION[1])
        pyautogui.click()
        pyautogui.write(method)

        pyautogui.moveTo(PopulateRows.__ROW_3_DATAFILE_POSITION[0],PopulateRows.__ROW_3_DATAFILE_POSITION[1])
        pyautogui.click()
        pyautogui.write(datafile)

class Scroller():
    __CLICKS = 1

    def __scroll():
        pyautogui.scroll(Scroller.__CLICKS)

class Runner():
    __RUNNER_BUTTON_POSITION = (0,0)

    def __run_worklist():
        pyautogui.moveTo(Runner.__RUNNER_BUTTON_POSITION[0],Runner.__RUNNER_BUTTON_POSITION[1])
        pyautogui.click()


# samples (key,value) is (sample_name,sample position)
def lipid_MRM(samples):
    METHOD = "lipid_MRM"
    METHODFILE = ""
    OUTPUT_PREFIX = ""
    if not isinstance(samples,dict):
        pass
    else:
        i = 1
        for sample,position in samples.items():
            if i == 1:
                PopulateRows.__populate_row_1(sample,position,METHODFILE,"{}/{}_{}".format(OUTPUT_PREFIX,sample,METHOD))
            elif i == 2: 
                PopulateRows.__populate_row_2(sample,position,METHODFILE,"{}/{}_{}".format(OUTPUT_PREFIX,sample,METHOD))
                Scroller.__scroll()
            elif (i%3) == 0:
                PopulateRows.__populate_row_1(sample,position,METHODFILE,"{}/{}_{}".format(OUTPUT_PREFIX,sample,METHOD))
            elif (i%3) == 1:
                PopulateRows.__populate_row_2(sample,position,METHODFILE,"{}/{}_{}".format(OUTPUT_PREFIX,sample,METHOD))
            elif (i%3) == 2:
                PopulateRows.__populate_row_3(sample,position,METHODFILE,"{}/{}_{}".format(OUTPUT_PREFIX,sample,METHOD))
                Scroller.__scroll()
            i+=1
        Runner.__run_worklist()