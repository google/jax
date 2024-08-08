# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
from numpy import array, float32


# Pasted from the test output (see back_compat_test_util.py module docstring)
data_2023_09_22 = dict(
    testdata_version=1,
    platform='tpu',
    custom_call_targets=['tpu_custom_call'],
    serialized_date=datetime.date(2023, 9, 22),
    inputs=(),
    expected_outputs=(array([[   90458.2  ,    90470.875,    90480.85 ,    90491.11 ,
           90500.945,    90510.95 ,    90521.18 ,    90530.95 ,
           90540.78 ,    90551.16 ,    90560.68 ,    90570.734,
           90580.73 ,    90590.58 ,    90600.66 ,    90610.61 ],
       [  643341.75 ,   643434.25 ,   643509.75 ,   643587.06 ,
          643660.1  ,   643735.9  ,   643813.5  ,   643886.   ,
          643960.6  ,   644039.56 ,   644110.25 ,   644186.75 ,
          644262.5  ,   644336.06 ,   644412.9  ,   644488.4  ],
       [ 1196323.2  ,  1196495.6  ,  1196636.8  ,  1196781.   ,
         1196917.5  ,  1197059.   ,  1197203.9  ,  1197339.2  ,
         1197478.5  ,  1197625.8  ,  1197757.8  ,  1197900.5  ,
         1198042.   ,  1198179.4  ,  1198323.   ,  1198464.   ],
       [ 1749075.5  ,  1749327.9  ,  1749534.4  ,  1749745.9  ,
         1749945.5  ,  1750152.8  ,  1750365.1  ,  1750563.1  ,
         1750767.1  ,  1750983.1  ,  1751176.2  ,  1751385.4  ,
         1751592.8  ,  1751793.8  ,  1752004.2  ,  1752210.8  ],
       [ 2302500.5  ,  2302832.5  ,  2303104.8  ,  2303383.5  ,
         2303646.2  ,  2303919.5  ,  2304199.   ,  2304459.8  ,
         2304728.5  ,  2305013.   ,  2305267.2  ,  2305543.   ,
         2305816.2  ,  2306081.   ,  2306358.5  ,  2306630.5  ],
       [ 2855440.2  ,  2855852.5  ,  2856190.2  ,  2856535.5  ,
         2856861.5  ,  2857200.5  ,  2857547.2  ,  2857870.5  ,
         2858204.5  ,  2858557.   ,  2858872.5  ,  2859214.5  ,
         2859553.2  ,  2859882.   ,  2860226.   ,  2860563.5  ],
       [ 3407472.   ,  3407964.2  ,  3408367.5  ,  3408780.2  ,
         3409169.5  ,  3409574.5  ,  3409988.5  ,  3410374.5  ,
         3410773.   ,  3411194.   ,  3411570.5  ,  3411979.   ,
         3412383.5  ,  3412776.   ,  3413186.5  ,  3413590.   ],
       [ 3959847.5  ,  3960419.   ,  3960888.   ,  3961367.8  ,
         3961820.2  ,  3962290.8  ,  3962772.5  ,  3963221.2  ,
         3963684.8  ,  3964174.2  ,  3964612.2  ,  3965086.8  ,
         3965557.2  ,  3966013.2  ,  3966491.   ,  3966959.5  ],
       [ 4515869.5  ,  4516521.5  ,  4517056.   ,  4517602.   ,
         4518118.   ,  4518654.5  ,  4519203.   ,  4519715.   ,
         4520243.   ,  4520801.   ,  4521300.   ,  4521841.   ,
         4522378.   ,  4522897.   ,  4523441.5  ,  4523975.5  ],
       [ 5061659.   ,  5062390.   ,  5062990.   ,  5063603.5  ,
         5064182.   ,  5064784.5  ,  5065401.   ,  5065975.   ,
         5066567.5  ,  5067194.   ,  5067754.   ,  5068362.   ,
         5068964.   ,  5069547.   ,  5070159.   ,  5070759.   ],
       [ 5621329.   ,  5622141.   ,  5622806.5  ,  5623487.5  ,
         5624129.5  ,  5624797.   ,  5625481.   ,  5626118.   ,
         5626775.   ,  5627470.5  ,  5628092.   ,  5628765.   ,
         5629433.5  ,  5630080.5  ,  5630758.5  ,  5631424.   ],
       [ 6172821.   ,  6173712.   ,  6174443.   ,  6175191.   ,
         6175896.   ,  6176630.   ,  6177381.   ,  6178080.5  ,
         6178803.   ,  6179566.   ,  6180248.5  ,  6180988.   ,
         6181722.   ,  6182432.5  ,  6183178.   ,  6183908.   ],
       [ 6723343.5  ,  6724315.   ,  6725111.5  ,  6725927.   ,
         6726696.   ,  6727495.5  ,  6728313.5  ,  6729076.5  ,
         6729863.5  ,  6730696.   ,  6731440.   ,  6732246.   ,
         6733046.   ,  6733820.5  ,  6734632.   ,  6735428.5  ],
       [ 7280537.   ,  7281587.5  ,  7282449.5  ,  7283331.5  ,
         7284163.5  ,  7285028.5  ,  7285914.   ,  7286739.5  ,
         7287591.   ,  7288492.   ,  7289296.5  ,  7290169.5  ,
         7291035.   ,  7291873.5  ,  7292752.5  ,  7293614.   ],
       [ 7828292.   ,  7829423.   ,  7830350.   ,  7831299.5  ,
         7832194.5  ,  7833125.5  ,  7834078.5  ,  7834966.   ,
         7835883.   ,  7836852.   ,  7837717.5  ,  7838657.   ,
         7839588.   ,  7840490.   ,  7841436.   ,  7842363.5  ],
       [ 8384808.5  ,  8386019.5  ,  8387012.5  ,  8388029.5  ,
         8388988.   ,  8389985.   ,  8391005.   ,  8391956.   ,
         8392937.   ,  8393974.   ,  8394902.   ,  8395907.   ,
         8396904.   ,  8397870.   ,  8398882.   ,  8399875.   ],
       [ 8928697.   ,  8929987.   ,  8931044.   ,  8932126.   ,
         8933146.   ,  8934208.   ,  8935294.   ,  8936306.   ,
         8937351.   ,  8938455.   ,  8939443.   ,  8940514.   ,
         8941574.   ,  8942604.   ,  8943682.   ,  8944738.   ],
       [ 9501496.   ,  9502866.   ,  9503990.   ,  9505141.   ,
         9506226.   ,  9507354.   ,  9508508.   ,  9509584.   ,
         9510695.   ,  9511870.   ,  9512919.   ,  9514058.   ,
         9515186.   ,  9516279.   ,  9517425.   ,  9518549.   ],
       [10055416.   , 10056868.   , 10058060.   , 10059279.   ,
        10060428.   , 10061624.   , 10062848.   , 10063988.   ,
        10065166.   , 10066410.   , 10067522.   , 10068729.   ,
        10069924.   , 10071083.   , 10072298.   , 10073489.   ],
       [10595886.   , 10597417.   , 10598673.   , 10599958.   ,
        10601170.   , 10602431.   , 10603721.   , 10604923.   ,
        10606164.   , 10607477.   , 10608649.   , 10609921.   ,
        10611182.   , 10612404.   , 10613684.   , 10614941.   ],
       [11135804.   , 11137412.   , 11138732.   , 11140083.   ,
        11141357.   , 11142682.   , 11144038.   , 11145301.   ,
        11146606.   , 11147985.   , 11149218.   , 11150554.   ,
        11151880.   , 11153163.   , 11154509.   , 11155829.   ],
       [11686791.   , 11688480.   , 11689864.   , 11691282.   ,
        11692618.   , 11694007.   , 11695430.   , 11696756.   ,
        11698123.   , 11699571.   , 11700864.   , 11702265.   ,
        11703656.   , 11705003.   , 11706414.   , 11707799.   ],
       [12263420.   , 12265190.   , 12266642.   , 12268128.   ,
        12269529.   , 12270986.   , 12272478.   , 12273868.   ,
        12275303.   , 12276820.   , 12278176.   , 12279646.   ,
        12281104.   , 12282516.   , 12283996.   , 12285447.   ],
       [12821178.   , 12823029.   , 12824548.   , 12826102.   ,
        12827567.   , 12829092.   , 12830652.   , 12832105.   ,
        12833606.   , 12835193.   , 12836610.   , 12838149.   ,
        12839673.   , 12841150.   , 12842699.   , 12844217.   ],
       [13362964.   , 13364895.   , 13366479.   , 13368100.   ,
        13369628.   , 13371218.   , 13372846.   , 13374362.   ,
        13375927.   , 13377582.   , 13379061.   , 13380665.   ,
        13382255.   , 13383796.   , 13385411.   , 13386995.   ],
       [13902882.   , 13904891.   , 13906539.   , 13908225.   ,
        13909815.   , 13911470.   , 13913163.   , 13914740.   ,
        13916369.   , 13918091.   , 13919629.   , 13921298.   ,
        13922953.   , 13924556.   , 13926236.   , 13927884.   ],
       [14443848.   , 14445934.   , 14447646.   , 14449398.   ,
        14451050.   , 14452769.   , 14454528.   , 14456166.   ,
        14457858.   , 14459647.   , 14461245.   , 14462979.   ,
        14464698.   , 14466363.   , 14468108.   , 14469820.   ],
       [15024407.   , 15026576.   , 15028355.   , 15030176.   ,
        15031893.   , 15033679.   , 15035507.   , 15037210.   ,
        15038969.   , 15040827.   , 15042490.   , 15044291.   ,
        15046077.   , 15047808.   , 15049621.   , 15051400.   ],
       [15586096.   , 15588347.   , 15590193.   , 15592082.   ,
        15593863.   , 15595716.   , 15597613.   , 15599380.   ,
        15601204.   , 15603133.   , 15604857.   , 15606726.   ,
        15608579.   , 15610375.   , 15612257.   , 15614103.   ],
       [16130043.   , 16132373.   , 16134285.   , 16136242.   ,
        16138087.   , 16140006.   , 16141970.   , 16143800.   ,
        16145690.   , 16147688.   , 16149473.   , 16151409.   ,
        16153328.   , 16155188.   , 16157138.   , 16159049.   ],
       [16669961.   , 16672369.   , 16674345.   , 16676367.   ,
        16678274.   , 16680257.   , 16682287.   , 16684178.   ,
        16686131.   , 16688196.   , 16690041.   , 16692042.   ,
        16694026.   , 16695948.   , 16697962.   , 16699938.   ],
       [17209878.   , 17212364.   , 17214404.   , 17216492.   ,
        17218460.   , 17220508.   , 17222604.   , 17224556.   ,
        17226572.   , 17228704.   , 17230608.   , 17232676.   ,
        17234724.   , 17236708.   , 17238788.   , 17240828.   ],
       [17817286.   , 17819860.   , 17821972.   , 17824132.   ,
        17826172.   , 17828292.   , 17830460.   , 17832482.   ,
        17834570.   , 17836776.   , 17838748.   , 17840888.   ,
        17843008.   , 17845062.   , 17847216.   , 17849328.   ],
       [18357204.   , 18359856.   , 18362032.   , 18364258.   ,
        18366358.   , 18368542.   , 18370778.   , 18372860.   ,
        18375012.   , 18377284.   , 18379316.   , 18381520.   ,
        18383704.   , 18385820.   , 18388040.   , 18390216.   ],
       [18897120.   , 18899852.   , 18902092.   , 18904384.   ,
        18906544.   , 18908794.   , 18911096.   , 18913240.   ,
        18915452.   , 18917792.   , 18919884.   , 18922152.   ,
        18924402.   , 18926580.   , 18928864.   , 18931104.   ],
       [19437040.   , 19439848.   , 19442152.   , 19444508.   ,
        19446732.   , 19449044.   , 19451412.   , 19453616.   ,
        19455894.   , 19458302.   , 19460452.   , 19462786.   ,
        19465100.   , 19467340.   , 19469688.   , 19471992.   ],
       [19976956.   , 19979844.   , 19982212.   , 19984634.   ,
        19986920.   , 19989296.   , 19991728.   , 19993996.   ,
        19996336.   , 19998810.   , 20001020.   , 20003420.   ,
        20005796.   , 20008100.   , 20010514.   , 20012882.   ],
       [20516874.   , 20519838.   , 20522270.   , 20524760.   ,
        20527106.   , 20529548.   , 20532046.   , 20534374.   ,
        20536776.   , 20539318.   , 20541588.   , 20544052.   ,
        20546492.   , 20548860.   , 20551338.   , 20553770.   ],
       [21056792.   , 21059834.   , 21062330.   , 21064884.   ,
        21067292.   , 21069798.   , 21072364.   , 21074752.   ,
        21077218.   , 21079826.   , 21082156.   , 21084684.   ,
        21087190.   , 21089618.   , 21092162.   , 21094658.   ],
       [21596710.   , 21599830.   , 21602390.   , 21605010.   ,
        21607480.   , 21610050.   , 21612680.   , 21615130.   ,
        21617660.   , 21620336.   , 21622724.   , 21625318.   ,
        21627888.   , 21630378.   , 21632988.   , 21635548.   ],
       [22218698.   , 22221906.   , 22224536.   , 22227228.   ,
        22229768.   , 22232408.   , 22235108.   , 22237628.   ,
        22240228.   , 22242976.   , 22245434.   , 22248094.   ,
        22250734.   , 22253292.   , 22255972.   , 22258602.   ],
       [22802946.   , 22806238.   , 22808938.   , 22811700.   ,
        22814306.   , 22817016.   , 22819790.   , 22822374.   ,
        22825044.   , 22827864.   , 22830386.   , 22833120.   ,
        22835830.   , 22838456.   , 22841208.   , 22843906.   ],
       [23351442.   , 23354816.   , 23357584.   , 23360416.   ,
        23363088.   , 23365866.   , 23368710.   , 23371360.   ,
        23374094.   , 23376988.   , 23379572.   , 23382374.   ,
        23385154.   , 23387846.   , 23390668.   , 23393436.   ],
       [23891360.   , 23894812.   , 23897644.   , 23900542.   ,
        23903274.   , 23906118.   , 23909028.   , 23911738.   ,
        23914536.   , 23917496.   , 23920140.   , 23923008.   ,
        23925850.   , 23928606.   , 23931492.   , 23934324.   ],
       [24431278.   , 24434808.   , 24437704.   , 24440668.   ,
        24443462.   , 24446368.   , 24449344.   , 24452116.   ,
        24454978.   , 24458004.   , 24460708.   , 24463640.   ,
        24466548.   , 24469364.   , 24472318.   , 24475214.   ],
       [24971196.   , 24974804.   , 24977764.   , 24980792.   ,
        24983648.   , 24986620.   , 24989662.   , 24992494.   ,
        24995420.   , 24998512.   , 25001276.   , 25004274.   ,
        25007244.   , 25010124.   , 25013142.   , 25016102.   ],
       [25511114.   , 25514800.   , 25517824.   , 25520918.   ,
        25523836.   , 25526872.   , 25529978.   , 25532872.   ,
        25535860.   , 25539020.   , 25541844.   , 25544906.   ,
        25547942.   , 25550884.   , 25553966.   , 25556990.   ],
       [26051032.   , 26054796.   , 26057884.   , 26061044.   ,
        26064022.   , 26067122.   , 26070296.   , 26073250.   ,
        26076302.   , 26079530.   , 26082412.   , 26085540.   ,
        26088640.   , 26091642.   , 26094792.   , 26097880.   ],
       [26590950.   , 26594790.   , 26597942.   , 26601168.   ,
        26604210.   , 26607374.   , 26610612.   , 26613628.   ,
        26616744.   , 26620038.   , 26622980.   , 26626172.   ,
        26629336.   , 26632402.   , 26635616.   , 26638768.   ],
       [27130866.   , 27134786.   , 27138002.   , 27141294.   ,
        27144396.   , 27147626.   , 27150930.   , 27154008.   ,
        27157186.   , 27160546.   , 27163548.   , 27166806.   ,
        27170034.   , 27173162.   , 27176440.   , 27179656.   ],
       [27723244.   , 27727248.   , 27730532.   , 27733892.   ,
        27737062.   , 27740358.   , 27743732.   , 27746876.   ,
        27750120.   , 27753552.   , 27756618.   , 27759944.   ,
        27763240.   , 27766436.   , 27769782.   , 27773064.   ],
       [28323220.   , 28327310.   , 28330664.   , 28334094.   ,
        28337330.   , 28340696.   , 28344142.   , 28347352.   ,
        28350664.   , 28354168.   , 28357300.   , 28360696.   ,
        28364062.   , 28367324.   , 28370744.   , 28374096.   ],
       [28885444.   , 28889618.   , 28893040.   , 28896544.   ,
        28899848.   , 28903284.   , 28906802.   , 28910078.   ,
        28913462.   , 28917038.   , 28920234.   , 28923702.   ,
        28927138.   , 28930468.   , 28933958.   , 28937382.   ],
       [29425518.   , 29429768.   , 29433256.   , 29436826.   ,
        29440192.   , 29443694.   , 29447276.   , 29450614.   ,
        29454062.   , 29457706.   , 29460962.   , 29464496.   ,
        29467996.   , 29471390.   , 29474946.   , 29478434.   ],
       [29965436.   , 29969764.   , 29973316.   , 29976952.   ,
        29980378.   , 29983944.   , 29987594.   , 29990992.   ,
        29994504.   , 29998214.   , 30001532.   , 30005128.   ,
        30008694.   , 30012148.   , 30015770.   , 30019322.   ],
       [30505352.   , 30509760.   , 30513376.   , 30517076.   ,
        30520566.   , 30524196.   , 30527910.   , 30531372.   ,
        30534944.   , 30538724.   , 30542100.   , 30545760.   ,
        30549392.   , 30552908.   , 30556594.   , 30560210.   ],
       [31045270.   , 31049756.   , 31053436.   , 31057202.   ,
        31060752.   , 31064446.   , 31068228.   , 31071750.   ,
        31075386.   , 31079232.   , 31082668.   , 31086394.   ,
        31090088.   , 31093668.   , 31097420.   , 31101100.   ],
       [31585188.   , 31589752.   , 31593496.   , 31597328.   ,
        31600940.   , 31604698.   , 31608544.   , 31612128.   ,
        31615828.   , 31619740.   , 31623236.   , 31627026.   ,
        31630786.   , 31634428.   , 31638244.   , 31641988.   ],
       [32125106.   , 32129748.   , 32133556.   , 32137452.   ,
        32141126.   , 32144950.   , 32148862.   , 32152506.   ,
        32156270.   , 32160248.   , 32163804.   , 32167660.   ,
        32171482.   , 32175186.   , 32179068.   , 32182876.   ],
       [32665024.   , 32669742.   , 32673614.   , 32677578.   ,
        32681314.   , 32685200.   , 32689178.   , 32692884.   ,
        32696710.   , 32700756.   , 32704372.   , 32708292.   ,
        32712180.   , 32715946.   , 32719894.   , 32723766.   ],
       [33221238.   , 33226038.   , 33229974.   , 33234004.   ,
        33237804.   , 33241756.   , 33245802.   , 33249570.   ,
        33253460.   , 33257576.   , 33261252.   , 33265238.   ,
        33269192.   , 33273022.   , 33277034.   , 33280972.   ],
       [33836944.   , 33841824.   , 33845832.   , 33849936.   ,
        33853804.   , 33857824.   , 33861940.   , 33865776.   ,
        33869736.   , 33873920.   , 33877664.   , 33881720.   ,
        33885744.   , 33889640.   , 33893724.   , 33897732.   ],
       [34414896.   , 34419864.   , 34423944.   , 34428112.   ,
        34432048.   , 34436140.   , 34440328.   , 34444232.   ,
        34448260.   , 34452520.   , 34456324.   , 34460456.   ,
        34464548.   , 34468512.   , 34472672.   , 34476744.   ],
       [34824696.   , 34829728.   , 34833856.   , 34838080.   ,
        34842064.   , 34846208.   , 34850448.   , 34854396.   ,
        34858476.   , 34862792.   , 34866644.   , 34870824.   ,
        34874968.   , 34878984.   , 34883192.   , 34887320.   ]],
      dtype=float32),),
    mlir_module_text=r"""
#loc4 = loc("third_party/py/jax_triton/googlexpallas_tpu/back_compat_test.py":33:0)
#loc11 = loc("jit(func)/jit(main)/pjit[in_shardings=(UnspecifiedValue, UnspecifiedValue) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False, False) name=matmul keep_unused=False inline=False]"(#loc4))
#loc16 = loc("jit(func)/jit(main)/jit(matmul)/pjit[in_shardings=(UnspecifiedValue, UnspecifiedValue) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False, False) name=wrapped keep_unused=False inline=False]"(#loc4))
#loc17 = loc("jit(func)/jit(main)/jit(matmul)/jit(wrapped)/pjit[in_shardings=(UnspecifiedValue, UnspecifiedValue) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False, False) name=apply_kernel keep_unused=False inline=False]"(#loc4))
module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<64x16xf32> {jax.result_info = ""}) {
    %0 = stablehlo.constant dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688, 704, 720, 736, 752, 768, 784, 800, 816, 832, 848, 864, 880, 896, 912, 928, 944, 960, 976, 992, 1008]> : tensor<64xi32> loc(#loc)
    %1 = stablehlo.constant dense<[0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]> : tensor<16xi32> loc(#loc)
    %2 = stablehlo.iota dim = 0 : tensor<524288xf32> loc(#loc6)
    %3 = stablehlo.reshape %2 : (tensor<524288xf32>) -> tensor<1024x512xf32> loc(#loc7)
    %4 = stablehlo.constant dense<1.000000e-03> : tensor<f32> loc(#loc)
    %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<1024x512xf32> loc(#loc8)
    %6 = stablehlo.multiply %5, %3 : tensor<1024x512xf32> loc(#loc8)
    %7 = stablehlo.constant dense<1.000000e+00> : tensor<f32> loc(#loc)
    %8 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<1024x512xf32> loc(#loc9)
    %9 = stablehlo.add %8, %6 : tensor<1024x512xf32> loc(#loc9)
    %10 = stablehlo.slice %9 [0:512, 0:256] : (tensor<1024x512xf32>) -> tensor<512x256xf32> loc(#loc10)
    %11 = call @matmul(%9, %10) : (tensor<1024x512xf32>, tensor<512x256xf32>) -> tensor<1024x256xf32> loc(#loc11)
    %12 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<64xi32>) -> tensor<64x16x1xi32> loc(#loc12)
    %13 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<16xi32>) -> tensor<64x16x1xi32> loc(#loc13)
    %14 = stablehlo.concatenate %12, %13, dim = 2 : (tensor<64x16x1xi32>, tensor<64x16x1xi32>) -> tensor<64x16x2xi32> loc(#loc14)
    %15 = "stablehlo.gather"(%11, %14) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, indices_are_sorted = true, slice_sizes = dense<1> : tensor<2xi64>} : (tensor<1024x256xf32>, tensor<64x16x2xi32>) -> tensor<64x16xf32> loc(#loc15)
    return %15 : tensor<64x16xf32> loc(#loc)
  } loc(#loc)
  func.func private @matmul(%arg0: tensor<1024x512xf32> loc("jit(func)/jit(main)/pjit[in_shardings=(UnspecifiedValue, UnspecifiedValue) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False, False) name=matmul keep_unused=False inline=False]"(#loc4)), %arg1: tensor<512x256xf32> loc("jit(func)/jit(main)/pjit[in_shardings=(UnspecifiedValue, UnspecifiedValue) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False, False) name=matmul keep_unused=False inline=False]"(#loc4))) -> tensor<1024x256xf32> {
    %0 = call @wrapped(%arg0, %arg1) : (tensor<1024x512xf32>, tensor<512x256xf32>) -> tensor<1024x256xf32> loc(#loc16)
    return %0 : tensor<1024x256xf32> loc(#loc11)
  } loc(#loc11)
  func.func private @wrapped(%arg0: tensor<1024x512xf32> loc("jit(func)/jit(main)/jit(matmul)/pjit[in_shardings=(UnspecifiedValue, UnspecifiedValue) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False, False) name=wrapped keep_unused=False inline=False]"(#loc4)), %arg1: tensor<512x256xf32> loc("jit(func)/jit(main)/jit(matmul)/pjit[in_shardings=(UnspecifiedValue, UnspecifiedValue) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False, False) name=wrapped keep_unused=False inline=False]"(#loc4))) -> tensor<1024x256xf32> {
    %0 = call @apply_kernel(%arg0, %arg1) : (tensor<1024x512xf32>, tensor<512x256xf32>) -> tensor<1024x256xf32> loc(#loc17)
    return %0 : tensor<1024x256xf32> loc(#loc16)
  } loc(#loc16)
  func.func private @apply_kernel(%arg0: tensor<1024x512xf32> loc("jit(func)/jit(main)/jit(matmul)/jit(wrapped)/pjit[in_shardings=(UnspecifiedValue, UnspecifiedValue) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False, False) name=apply_kernel keep_unused=False inline=False]"(#loc4)), %arg1: tensor<512x256xf32> loc("jit(func)/jit(main)/jit(matmul)/jit(wrapped)/pjit[in_shardings=(UnspecifiedValue, UnspecifiedValue) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False, False) name=apply_kernel keep_unused=False inline=False]"(#loc4))) -> tensor<1024x256xf32> {
    %0 = stablehlo.custom_call @tpu_custom_call(%arg0, %arg1) {backend_config = "{\22custom_call_config\22: {\22body\22: \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MDEwMDsgIDAxMDA7oCAwMTAwPCAgMDEwMDygIDAxMDA9ICAwMTAwPaAgMDEwMD4gIDAxMDA+oCAwMTAwPyAgMDEwMN+gIDAREGDwMbAwURBg8DHQMHEQYPAx8DCQcHAwEDAQeNiYkHBwMBAwEHjYmFBwcDAQMBB42DiQcHAwEDAQeNg4UHBwMBAwEHjYGJBwcDAQMBB42BhQcHAwEDAQeNf4kHBwMBAwEHjX+FBwcDAQMBB419iQcHAwEDAQeNfYUHBwMBAwEHjXuJBwcDAQMBB417hQcHAwEDAQeNeYkHBwMBAwEHjXmFBwcDAQMBB413iQcHAwEDAQeNd4UHBwMBAwEHjXWJBwcDAQMBB411hQcHAwEDAQeNc4kHBwMBAwEHjXOFBwcDAQMBB41xiQcHAwEDAQeNcYUHBwMBAwEHjW+JBwcDAQMBB41vhQcHAwEDAQeNbYkHBwMBAwEHjW2FBwcDAQMBB41riQcHAwEDAQeNa4UHBwMBAwEHjWmJBwcDAQMBB41phQcHAwEDAQeNZ4kHBwMBAwEHjWeFBwcDAQMBB42FiQcHAwEDAQeNhYUHBwMBAwEHjWWJBwcDAQMBB41lhQcHAwEDAQeNY4kHBwMBAwEHjWOFBwcDAQMBB41hiQcHAwEDAQeNYYUHBwMBAwEHjV+JBwcDAQMBB41fhQcHAwEDAQeNXYkHBwMBAwEHjV2FBwcDAQMBB41biQcHAwEDAQeNW4UHBwMBAwEHjVmJBwcDAQMBB41ZhQcHAwEDAQeNV4kHBwMBAwEHjVeFBwcDAQMBB41ViQcHAwEDAQeNVYUHBwMBAwEHjVOJBwcDAQMBB41ThQcHAwEDAQeNUYkHBwMBAwEHjVGFBwcDAQMBB41PiQcHAwEDAQeNT4UHBwMBAwEHjU2JBwcDAQMBB41NhQcHAwEDAQeNS4kHBwMBAwEHjUuFBwcDAQMBB41JiQcHAwEDAQeNSYUHBwUBAwEHj4mJBwcFAQMBB4+JhQcHBQEDAQePg4kHBwUBAwEHj4OFBwcFAQMBB4+BiQcHBQEDAQePgYUHBwUBAwEHj3+JBwcFAQMBB49/hQcHBQEDAQePfYkHBwUBAwEHj32FBwcFAQMBB497iQcHBQEDAQePe4UHBwUBAwEHj3mJBwcFAQMBB495hQcHBQEDAQePd4kHBwUBAwEHj3eFBwcFAQMBB491iQcHBQEDAQePdYUHBwUBAwEHj3OJBwcFAQMBB49zhQcHBQEDAQePcYkHBwUBAwEHj3GFBwcFAQMBB49viQcHBQEDAQePb4UHBwUBAwEHj22JBwcFAQMBB49thQcHBQEDAQePa4kHBwUBAwEHj2uFBwcFAQMBB49piQcHBQEDAQePaYUHBwUBAwEHj2eJBwcFAQMBB49nhQcHBQEDAQePhYkHBwUBAwEHj4WFBwcFAQMBB49liQcHBQEDAQePZYUHBwUBAwEHj2OJBwcFAQMBB49jhQcHBQEDAQePYYkHBwUBAwEHj2GFBwcFAQMBB49fiQcHBQEDAQePX4UHBwUBAwEHj12JBwcFAQMBB49dhQcHBQEDAQePW4kHBwUBAwEHj1uFBwcFAQMBB49ZiQcHBQEDAQePWYUHBwUBAwEHj1eJBwcFAQMBB49XhQcHBQEDAQePVYkHBwUBAwEHj1WFBwcFAQMBB49TiQcHBQEDAQePU4UHBwUBAwEHj1GJBwcFAQMBB49RhQcHBQEDAQePT4kHBwUBAwEHj0+FBwcFAQMBB49NiQcHBQEDAQePTYUHBwUBAwEHj0uJBwcFAQMBB49LhQcHBQEDAQePSYkHBwUBAwEHj0mFBwcDAQMBB42JhwcHAwEDAQeNiUcHBwMBAwEHjYOHBwcDAQMBB42DRwcHAwEDAQeNgYcHBwMBAwEHjYFHBwcDAQMBB41/hwcHAwEDAQeNf0cHBwMBAwEHjX2HBwcDAQMBB419RwcHAwEDAQeNe4cHBwMBAwEHjXtHBwcDAQMBB415hwcHAwEDAQeNeUcHBwMBAwEHjXeHBwcDAQMBB413RwcHAwEDAQeNdYcHBwMBAwEHjXVHBwcDAQMBB41zhwcHAwEDAQeNc0cHBwMBAwEHjXGHBwcDAQMBB41xRwcHAwEDAQeNb4cHBwMBAwEHjW9HBwcDAQMBB41thwcHAwEDAQeNbUcHBwMBAwEHjWuHBwcDAQMBB41rRwcHAwEDAQeNaYcHBwMBAwEHjWlHBwcDAQMBB41nhwcHAwEDAQeNZ0cHBwMBAwEHjYWHBwcDAQMBB42FRwcHAwEDAQeNZYcHBwMBAwEHjWVHBwcDAQMBB41jhwcHAwEDAQeNY0cHBwMBAwEHjWGHBwcDAQMBB41hRwcHAwEDAQeNX4cHBwMBAwEHjV9HBwcDAQMBB41dhwcHAwEDAQeNXUcHBwMBAwEHjVuHBwcDAQMBB41bRwcHAwEDAQeNWYcHBwMBAwEHjVlHBwcDAQMBB41XhwcHAwEDAQeNV0cHBwMBAwEHjVWHBwcDAQMBB41VRwcHAwEDAQeNU4cHBwMBAwEHjVNHBwcDAQMBB41RhwcHAwEDAQeNUUcHBwMBAwEHjU+HBwcDAQMBB41PRwcHAwEDAQeNTYcHBwMBAwEHjU1HBwcDAQMBB41LhwcHAwEDAQeNS0cHBwMBAwEHjUmHBwcDAQMBB41JRwcHBQEDAQePh4kHBwUBAwEHj4eFBwcFAQMBB49FiQcHBQEDAQePRYUHBwUBAwEHj0OJBwcFAQMBB49DhQcHBQEDAQePQYkHBwUBAwEHj0GFBwcFAQMBB48/iQcHBQEDAQePP4UHBwUBAwEHjz2JBwcFAQMBB489hQcHBQEDAQePO4kHBwUBAwEHjzuFBwcFAQMBB485iQcHBQEDAQePOYUHBwUBAwEHjzeJBwcFAQMBB483hQcHBQEDAQePNYkHBwUBAwEHjzWFBwcFAQMBB48ziQcHBQEDAQePM4UHBwUBAwEHjzGJBwcFAQMBB48xhQcHBQEDAQePL4kHBwUBAwEHjy+FBwcFAQMBB48tiQcHBQEDAQePLYUHBwUBAwEHjyuJBwcFAQMBB48rhQcHBQEDAQePKYkHBwUBAwEHjymFBwcFAQMBB49HiQcHBQEDAQePR4UHBwUBAwEHjyeJBwcFAQMBB48nhQcHBQEDAQePJYkHBwUBAwEHjyWFBwcFAQMBB48jiQcHBQEDAQePI4UHBwUBAwEHjyGJBwcFAQMBB48hhQcHBQEDAQePH4kHBwUBAwEHjx+FBwcFAQMBB48diQcHBQEDAQePHYUHBwUBAwEHjxuJBwcFAQMBB48bhQcHBQEDAQePGYkHBwUBAwEHjxmFBwcFAQMBB48XiQcHBQEDAQePF4UHBwUBAwEHjxWJBwcFAQMBB48VhQcHBQEDAQePE4kHBwUBAwEHjxOFBwcFAQMBB48RiQcHBQEDAQePEYUHBwUBAwEHjw+JBwcFAQMBB48PhQcHBQEDAQePDYkHBwUBAwEHjw2FBwcFAQMBB48LiQcHBQEDAQePC4ULBw0RAwVBJgMuAzYDPgNGA04DVgNeA2YDbgN2A34DhgOOA5YDngOmA64DtgO+A8YDzgPWA94D5gPuA/YD/gMGBA4EFgQeBAsHDREDBUEqAzIDOgNCA0oDUgNaA2IDagNyA3oDggOKA5IDmgOiA6oDsgO6A8IDygPSA9oD4gPqA/ID+gMCBAoEEgQaBCIECwcNEQMLISYELgQ2BD4ERgROBFYEXgRmBG4EdgR+BIYEjgSWBJ4ECwcNEQMFQYuLi4uLi4uLi4uLi4uLi4uLi4uLi4uLi4uLi4uLi4uLDQcNEwMFByYFLgUyBQ8HDRVBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEDNgULBw0RAwshpgSuBLYEvgTGBM4E1gTeBOYE7gT2BP4EBgUOBRYFHgULBw0RAwVBOgU+BUIFRgVKBU4FUgVWBVoFXgViBWYFagVuBXIFdgV6BX4FggWGBYoFjgWSBZYFmgWeBaIFpgWqBa4FsgW2BQ0HDRMDBQcqBboFvgUPBw0VQQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA8IFCwcNEQMLISoEMgQ6BEIESgRSBFoEYgRqBHIEegSCBIoEkgSaBKIECwcNEQMFQYuLi4uLi4uLi4uLi4uLi4uLi4uLi4uLi4uLi4uLi4uLDQcNEwMFByYFRgZKBg8HDRVBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEDTgYLBw0RAwshqgSyBLoEwgTKBNIE2gTiBOoE8gT6BAIFCgUSBRoFIgULBw0RAwVBUgZWBloGXgZiBmYGagZuBnIGdgZ6Bn4GggaGBooGjgaSBpYGmgaeBqIGpgaqBq4Gsga2BroGvgbCBsYGygbOBg0HDRMDBQcqBdIG1gYPBw0VQQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA9oGCwcNEQMFQZOXm5+jp6uvs7e7v8PHy8/T19vf4+fr7/P3+/8GAg4CFgIeAgsHDREDBUGVmZ2hpamtsbW5vcHFyc3R1dnd4eXp7fH1+f0CAgoCEgIaAiICCwcNEQMLISYCLgI2Aj4CRgJOAlYCXgJmAm4CdgJ+AoYCjgKWAp4CCwcNEQMFQcYFygXOBdIF1gXaBd4F4gXmBeoF7gXyBfYF+gX+BQIGBgYKBg4GEgYWBhoGHgYiBiYGKgYuBjIGNgY6Bj4GQgYNBw0TAwUHXgdmB2oHDwcNFUEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQNuBwsHDREDCyGmAq4CtgK+AsYCzgLWAt4C5gLuAvYC/gIGAw4DFgMeAwsHDREDBUFyB3YHegd+B4IHhgeKB44HkgeWB5oHngeiB6YHqgeuB7IHtge6B74HwgfGB8oHzgfSB9YH2gfeB+IH5gfqB+4HDQcNEwMFB2IH8gf2Bw8HDRVBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQED+gcLBw0RAwshKgIyAjoCQgJKAlICWgJiAmoCcgJ6AoICigKSApoCogILBw0RAwVB3gbiBuYG6gbuBvIG9gb6Bv4GAgcGBwoHDgcSBxYHGgceByIHJgcqBy4HMgc2BzoHPgdCB0YHSgdOB1IHVgdaBw0HDRMDBQdeB34IgggPBw0VQQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA4YICwcNEQMLIaoCsgK6AsICygLSAtoC4gLqAvIC+gICAwoDEgMaAyIDCwcNEQMFQYoIjgiSCJYImgieCKIIpgiqCK4Isgi2CLoIvgjCCMYIygjOCNII1gjaCN4I4gjmCOoI7gjyCPYI+gj+CAIJBgkNBw0TAwUHYgcKCQ4JDwcNFUEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQMSCQkFCwkJ/geRiYkJBQsJCRYJkYmFCQULCQkCCJGDiQkFCwkJGgmRg4UJBQsJCQYIkYGJCQULCQkeCZGBhQkFCwkJCgiRf4kJBQsJCSIJkX+FCQULCQkOCJF9iQkFCwkJJgmRfYUJBQsJCRIIkXuJCQULCQkqCZF7hQkFCwkJFgiReYkJBQsJCS4JkXmFCQULCQkaCJF3iQkFCwkJMgmRd4UJBQsJCR4IkXWJCQULCQk2CZF1hQkFCwkJIgiRc4kJBQsJCToJkXOFCQULCQkmCJFxiQkFCwkJPgmRcYUJBQsJCSoIkW+JCQULCQlCCZFvhQkFCwkJLgiRbYkJBQsJCUYJkW2FCQULCQkyCJFriQkFCwkJSgmRa4UJBQsJCTYIkWmJCQULCQlOCZFphQkFCwkJOgiRZ4kJBQsJCVIJkWeFCQULCQk+CJGFiQkFCwkJVgmRhYUJBQsJCUIIkWWJCQULCQlaCZFlhQkFCwkJRgiRY4kJBQsJCV4JkWOFCQULCQlKCJFhiQkFCwkJYgmRYYUJBQsJCU4IkV+JCQULCQlmCZFfhQkFCwkJUgiRXYkJBQsJCWoJkV2FCQULCQlWCJFbiQkFCwkJbgmRW4UJBQsJCVoIkVmJCQULCQlyCZFZhQkFCwkJXgiRV4kJBQsJCXYJkVeFCQULCQliCJFViQkFCwkJegmRVYUJBQsJCWYIkVOJCQULCQl+CZFThQkFCwkJagiRUYkJBQsJCYIJkVGFCQULCQluCJFPiQkFCwkJhgmRT4UJBQsJCXIIkU2JCQULCQmKCZFNhQkFCwkJdgiRS4kJBQsJCY4JkUuFCQULCQl6CJFJiQkFCwkJkgmRSYUFAQ8eAwMRD2UHAwcLBQcPBw8TAw8vAwcFBA8FAQUDEQ9nBwMHCwUHDwcPEwMPLwMHBQQPBQUDAxEPaQcDBQcFBw8HDwUEDwUBAwYDAQUBAE4VTXYDKR0dF04C/gOB/gMdCyEjKR8bGRkZHSUTHRUNEykfDxsNCw8PDQkLEWJ1aWx0aW4AZnVuYwB0cHUAYXJpdGgAbW9kdWxlAHJldHVybgBsb2FkAHN0b3JlAHJvbGxfdmVjdG9ycwBtYXRtdWwAdW5yb2xsX3ZlY3RvcnMAZXJhc2VfbWVtcmVmX2xheW91dABjb25zdGFudAB2YWx1ZQBpbl9sYXlvdXQAZnVuY3Rpb25fdHlwZQBzeW1fbmFtZQB0cmFuc2Zvcm1faW5kaWNlcwB3aW5kb3dfYm91bmRzAHRyYW5zZm9ybV8wAHRyYW5zZm9ybV8xAHRyYW5zZm9ybV8yAHN1YmxhbmVfbWFzawBzdWJsYW5lX3N0cmlkZQBkaW1lbnNpb25fc2VtYW50aWNzAGl0ZXJhdGlvbl9ib3VuZHMAc2NhbGFyX3ByZWZldGNoAG1haW4Ad2luZG93X3BhcmFtcwAvbWFza2VkX2xvYWRbbWFza2VkPUZhbHNlIGNhY2hlX21vZGlmaWVyPSBldmljdGlvbl9wb2xpY3k9IGlzX3ZvbGF0aWxlPUZhbHNlIGFyZ3NfdHJlZT1QeVRyZWVEZWYoKEN1c3RvbU5vZGUoTkRJbmRleGVyWyhbXSwgUHlUcmVlRGVmKFtDdXN0b21Ob2RlKFNsaWNlWyhGYWxzZSwgMjU2KV0sIFsqXSksIEN1c3RvbU5vZGUoU2xpY2VbKFRydWUsIDAsIDI1NildLCBbXSldKSwgW1RydWUsIFRydWVdLCAoNTEyLCAyNTYpLCAoKSldLCBbKl0pLCkpXQB0aGlyZF9wYXJ0eS9weS9qYXhfdHJpdG9uL2dvb2dsZS9wYWxsYXNfdHB1L2JhY2tfY29tcGF0X3Rlc3QucHkAL21hc2tlZF9sb2FkW21hc2tlZD1GYWxzZSBjYWNoZV9tb2RpZmllcj0gZXZpY3Rpb25fcG9saWN5PSBpc192b2xhdGlsZT1GYWxzZSBhcmdzX3RyZWU9UHlUcmVlRGVmKChDdXN0b21Ob2RlKE5ESW5kZXhlclsoW10sIFB5VHJlZURlZihbQ3VzdG9tTm9kZShTbGljZVsoVHJ1ZSwgMCwgMjU2KV0sIFtdKSwgQ3VzdG9tTm9kZShTbGljZVsoRmFsc2UsIDI1NildLCBbKl0pXSksIFtUcnVlLCBUcnVlXSwgKDI1NiwgNTEyKSwgKCkpXSwgWypdKSwpKV0AL2RvdF9nZW5lcmFsW2RpbWVuc2lvbl9udW1iZXJzPSgoKDEsKSwgKDAsKSksICgoKSwgKCkpKSBwcmVjaXNpb249KDxQcmVjaXNpb24uREVGQVVMVDogMD4sIDxQcmVjaXNpb24uREVGQVVMVDogMD4pIHByZWZlcnJlZF9lbGVtZW50X3R5cGU9ZmxvYXQzMl0Ab3V0X2xheW91dAB0cmFuc3Bvc2VfbGhzAHRyYW5zcG9zZV9yaHMAb3BlcmFuZFNlZ21lbnRTaXplcwAvbWFza2VkX3N3YXBbbWFza2VkPUZhbHNlIGV2aWN0aW9uX3BvbGljeT0gYXJnc190cmVlPVB5VHJlZURlZigoQ3VzdG9tTm9kZShOREluZGV4ZXJbKFtdLCBQeVRyZWVEZWYoW0N1c3RvbU5vZGUoU2xpY2VbKFRydWUsIDAsIDI1NildLCBbXSksIEN1c3RvbU5vZGUoU2xpY2VbKFRydWUsIDAsIDI1NildLCBbXSldKSwgW1RydWUsIFRydWVdLCAoMjU2LCAyNTYpLCAoKSldLCBbXSksKSldAA==\22}}", kernel_name = "func", operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : (tensor<1024x512xf32>, tensor<512x256xf32>) -> tensor<1024x256xf32> loc(#loc18)
    return %0 : tensor<1024x256xf32> loc(#loc17)
  } loc(#loc17)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("third_party/py/jax_triton/googlexpallas_tpu/back_compat_test.py":30:0)
#loc2 = loc("third_party/py/jax_triton/googlexpallas_tpu/back_compat_test.py":31:0)
#loc3 = loc("third_party/py/jax_triton/googlexpallas_tpu/back_compat_test.py":32:0)
#loc5 = loc("third_party/py/jax_triton/googlexpallas_tpu/back_compat_test.py":35:0)
#loc6 = loc("jit(func)/jit(main)/iota[dtype=float32 shape=(524288,) dimension=0]"(#loc1))
#loc7 = loc("jit(func)/jit(main)/reshape[new_sizes=(1024, 512) dimensions=None]"(#loc2))
#loc8 = loc("jit(func)/jit(main)/mul"(#loc1))
#loc9 = loc("jit(func)/jit(main)/add"(#loc1))
#loc10 = loc("jit(func)/jit(main)/slice[start_indices=(0, 0) limit_indices=(512, 256) strides=None]"(#loc3))
#loc12 = loc("jit(func)/jit(main)/broadcast_in_dim[shape=(64, 16, 1) broadcast_dimensions=(0,)]"(#loc5))
#loc13 = loc("jit(func)/jit(main)/broadcast_in_dim[shape=(64, 16, 1) broadcast_dimensions=(1,)]"(#loc5))
#loc14 = loc("jit(func)/jit(main)/concatenate[dimension=2]"(#loc5))
#loc15 = loc("jit(func)/jit(main)/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0, 1), start_index_map=(0, 1)) slice_sizes=(1, 1) unique_indices=True indices_are_sorted=True mode=GatherScatterMode.PROMISE_IN_BOUNDS fill_value=None]"(#loc5))
#loc18 = loc("jit(func)/jit(main)/jit(matmul)/jit(wrapped)/jit(apply_kernel)/tpu_custom_call[config=CustomCallBackendConfig(<omitted>) kernel_name=func kernel_regeneration_metadata=None out_avals=(ShapedArray(float32[1024,256]),)]"(#loc4))
""",
    mlir_module_serialized=b'ML\xefR\x01StableHLO_v0.9.0\x00\x01+\x05\x01\x03\x01\x03\x05\x03\x1b\x07\t\x0b\r\x0f\x11\x13\x15\x17\x19\x1b\x1d\x1f\x03~\x02\xfb-\x01\xaf\x07\x0b\x0f\x0b\x0f\x0f\x0b\x0b\x0b\x0b\x13\x0b\x13\x0b\x13\x0b\x0f\x13\x0f\x0f+\x0b\x0f\x0b\x0b\x0b33\x0b3\x0b3\x0bS\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x13\x13\x13\x13\x13\x0b\x0f\x0b\x0f\x0b\x13\x13\x0b\x13\x0b#\x0b\x0b\x0b\x0f\x0b\x13\x13\x13\x0f\x0b\x13\x0f\x0b\x13\x0b\x0f\x0b;\x0b\x0b\x0b\x0b\x0b\x0b\x0f\x0b\x03M\x0b\x13\x0b\x0b\x0f\x0bO\x0b\x0b\x0b\x0f/\x0fO\x0b\x0f\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x13\x0f&\x08\x1e\x02\x0f\x1f\x1fO///\x0b\x01\x05\x0b\x0f\x03)\x1f\x07\x1f\x1f\x07\x07\x0f\x13\x1b\x17\x13\x1f\x13\x13\x1b\x13\x07\x1b\x13\x1f\x022\x0e\x1f\x05!\x1d9\x15\x05#\x1d=\x15\x1dA\x15\x05%\x05\'\x05)\x05+\x17\x07C\x01\x05-\x17\x07G\x01\x05/\x17\x07=\x01\x051\x11\x03\x05\x03\x03\x1f\xc3\x1ds\x1d\x1dw\x1d\x03\t+-/!1!\x033\x053\x11\x01\x00\x055\x057\x059\x03\x0b\r\xaf\x0f\xcb\x11\xcd\x03\xd5\x13\xd7\x03\x0b\r\xb1\x0f\xb5\x11\xb7\x03\xbd\x13\xb9\x05;\x03\x0b\r\xb1\x0f\xb5\x11\xb7\x03\xbf\x13\xb9\x05=\x03\x0b\r\xb1\x0f\xb5\x11\xb7\x03\xc1\x13\xb9\x05?\x03\x13E\xd9G\xdbI\xddK\xafM\xdfO\xe1Q\xe3S\xafU\xe5\x05A\x05C\x05E\x05G\x05I\x05K\x05M\x05O\x05Q\x1dY\x15\x05S\x03\x03\x1b\xc1\x03\x03\x1b\xbf\x03\x03\x17\xe7\x03\x03\x17\xe9\x03\x03e\xeb\x05U\x1di\x1d\x05W\x1dmo\x05Y\x17\x07?\x01\x03\x03\x17\xed\x05[\x03\x03\x17\xef\x05]\x03\x07{\xf1}\xf3\x7f\xc5\x05_\x05a\x05c\x1d\x83\x85\x05e\x17\x07A\x01\x03\x03\x1b\xbd\x03\x03\x1f\xf5\x1d\x8d\x19\x05g\x03\x03\x1f\xf7\x1d\x93\x19\x05i\x03\x03\x97\xc7\x05k\x1d\x9b\x19\x05m\x03\r\x9f\xc9\xa1\xc7\xa3\xf9\xa5\xc3\xa7\xc5\xa9\xc9\x05o\x05q\x05s\x05u\x05w\x05y\x1d\xad\x19\x05{\x03\x01\x03\x05\xb3\xb3\r\x01#!\x03\x03\xb3\x1d}\x1f#!\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x7f\x1d\x81\x1d\x83\x1f)\x01\x1f\x13\x11\x01\x00\x00\x00\x00\x00\x00\x00\x13\r\t\x1f\x13!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00#\x1f\x03\x03\xcf\r\x03\xd1\xd3\x1d\x85\x1d\x87\x1d\x89\x1d\x8b\x0b\x03\x1d\x8d\x1d\x8f\x05\x01\x1d\x91\x03\x05\xbb\xbb\x03\x03\xbb\x1f\x17\x02\x04\x00\x00\x00\x00\x10\x00\x00\x00 \x00\x00\x000\x00\x00\x00@\x00\x00\x00P\x00\x00\x00`\x00\x00\x00p\x00\x00\x00\x80\x00\x00\x00\x90\x00\x00\x00\xa0\x00\x00\x00\xb0\x00\x00\x00\xc0\x00\x00\x00\xd0\x00\x00\x00\xe0\x00\x00\x00\xf0\x00\x00\x00\x00\x01\x00\x00\x10\x01\x00\x00 \x01\x00\x000\x01\x00\x00@\x01\x00\x00P\x01\x00\x00`\x01\x00\x00p\x01\x00\x00\x80\x01\x00\x00\x90\x01\x00\x00\xa0\x01\x00\x00\xb0\x01\x00\x00\xc0\x01\x00\x00\xd0\x01\x00\x00\xe0\x01\x00\x00\xf0\x01\x00\x00\x00\x02\x00\x00\x10\x02\x00\x00 \x02\x00\x000\x02\x00\x00@\x02\x00\x00P\x02\x00\x00`\x02\x00\x00p\x02\x00\x00\x80\x02\x00\x00\x90\x02\x00\x00\xa0\x02\x00\x00\xb0\x02\x00\x00\xc0\x02\x00\x00\xd0\x02\x00\x00\xe0\x02\x00\x00\xf0\x02\x00\x00\x00\x03\x00\x00\x10\x03\x00\x00 \x03\x00\x000\x03\x00\x00@\x03\x00\x00P\x03\x00\x00`\x03\x00\x00p\x03\x00\x00\x80\x03\x00\x00\x90\x03\x00\x00\xa0\x03\x00\x00\xb0\x03\x00\x00\xc0\x03\x00\x00\xd0\x03\x00\x00\xe0\x03\x00\x00\xf0\x03\x00\x00\x1f\x19\x81\x00\x00\x00\x00\x10\x00\x00\x00 \x00\x00\x000\x00\x00\x00@\x00\x00\x00P\x00\x00\x00`\x00\x00\x00p\x00\x00\x00\x80\x00\x00\x00\x90\x00\x00\x00\xa0\x00\x00\x00\xb0\x00\x00\x00\xc0\x00\x00\x00\xd0\x00\x00\x00\xe0\x00\x00\x00\xf0\x00\x00\x00\x13\r\x01\x1f\x11\to\x12\x83:\x1f\x11\t\x00\x00\x80?\x1f\x13!\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x1f\x13\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x1d\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x1d\x11\x01\x00\x00\x00\x00\x00\x00\x00\x05\x03\x01\t\x01\x02\x02)\x05\x02 \x02\x10\x07\t)\x05\x02\x10\x02\x08\x07)\x05\x02 \x02\x08\x07\x1d\x1b)\x01\x07)\x03\t\r)\x05\x02\x02A\x07)\x03\x02\x02\x0f)\x03A\x0f)\x07\x02\x02A\x05\x0f)\x03\x05\r\x11\x01\x03\x15\x11\x05\x05\t\x03\x0b)\x03\t%\x13)\x03\x04\x00\x80\x07)\x03\x01\r)\x07\x02\x02A\t\x0f\x04~\x03\x05\x01\x11\x01)\x07\x03\x01\x11\x03\x11\x015\x07\x03!E\x07\x03\x01_\x03\x17\x07\x03\x01a\x03\x19\x0f\x03gc\x03\'\x11\x06k\x03\x05\x03\x05\x07\x03\x01q\x03\x11\t\x07%#\x03\x05\x03\t\x13\x06%\x03\x05\x05\x0b\x07\x07\x03\x01u\x03\x11\t\x07\'#\x03\x05\x03\x0f\x15\x06\'\x03\x05\x05\x11\r\x17\x07\x81y\x03\t\x03\x13\x0b\x07\x05\x87\x03\x0b\x05\x13\x15\t\x07\x8b\x89\x03\x1b\x03\x01\t\x07\x91\x8f\x03\x1b\x03\x03\x19\x07\x99\x95\x03+\x05\x19\x1b\x1b\x07\xab\x9d\x03\x15\x05\x17\x1d\x05\x04\x01\x03\x1f\x03\x11\x057\x07\x03\x07\x0b\x05\x05\x05\t\x05\x0b\x07\t]\x03\x0b\x05\x01\x03\x05\x04\x05\x03\x05\x03\x11\t;\x07\x03\x07\x0b\x05\x05\t\t\t\x0b\x07\x0b[\x03\x0b\x05\x01\x03\x05\x04\t\x03\x05\x03\x11\x0b?\x07\x03\x07\x0b\x05\x05\x0b\t\x0b\r\x07WC\x03\x0b\x05\x01\x03\x05\x04\x0b\x03\x05\x06\x03\x01\x05\x01\x00\xee\xcd\x93\x0b!f\xa7\x0f\x0b\x03!\x1b\x11\x0f\x11\n\x04!\x19\x19\'#+[\x15\xa5\xa5\xad\x11\x1d\x1d11\x87\x89\x1ff\x03\x1f/!\x19!)#\x1f\x19\xa2\x03Z\x03&\x03\x13%)9+\x0f\r\x1f\x15\x1d\x15\x81\x13\x15\x1f\x13\x0f\x19\x17\x11\x1f\x11)\x19\x15\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00func_v1\x00return_v1\x00constant_v1\x00broadcast_in_dim_v1\x00call_v1\x00custom_call_v1\x00iota_v1\x00reshape_v1\x00multiply_v1\x00add_v1\x00slice_v1\x00concatenate_v1\x00gather_v1\x00sym_name\x00third_party/py/jax_triton/googlexpallas_tpu/back_compat_test.py\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00value\x00callee\x00broadcast_dimensions\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_func\x00jit(func)/jit(main)/pjit[in_shardings=(UnspecifiedValue, UnspecifiedValue) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False, False) name=matmul keep_unused=False inline=False]\x00jit(func)/jit(main)/jit(matmul)/pjit[in_shardings=(UnspecifiedValue, UnspecifiedValue) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False, False) name=wrapped keep_unused=False inline=False]\x00jit(func)/jit(main)/jit(matmul)/jit(wrapped)/pjit[in_shardings=(UnspecifiedValue, UnspecifiedValue) out_shardings=(UnspecifiedValue,) resource_env=None donated_invars=(False, False) name=apply_kernel keep_unused=False inline=False]\x00api_version\x00backend_config\x00call_target_name\x00called_computations\x00has_side_effect\x00kernel_name\x00operand_layouts\x00output_operand_aliases\x00result_layouts\x00jit(func)/jit(main)/jit(matmul)/jit(wrapped)/jit(apply_kernel)/tpu_custom_call[config=CustomCallBackendConfig(<omitted>) kernel_name=func kernel_regeneration_metadata=None out_avals=(ShapedArray(float32[1024,256]),)]\x00iota_dimension\x00jit(func)/jit(main)/iota[dtype=float32 shape=(524288,) dimension=0]\x00jit(func)/jit(main)/reshape[new_sizes=(1024, 512) dimensions=None]\x00jit(func)/jit(main)/mul\x00jit(func)/jit(main)/add\x00limit_indices\x00start_indices\x00strides\x00jit(func)/jit(main)/slice[start_indices=(0, 0) limit_indices=(512, 256) strides=None]\x00jit(func)/jit(main)/broadcast_in_dim[shape=(64, 16, 1) broadcast_dimensions=(0,)]\x00jit(func)/jit(main)/broadcast_in_dim[shape=(64, 16, 1) broadcast_dimensions=(1,)]\x00dimension\x00jit(func)/jit(main)/concatenate[dimension=2]\x00collapsed_slice_dims\x00index_vector_dim\x00indices_are_sorted\x00offset_dims\x00slice_sizes\x00start_index_map\x00jit(func)/jit(main)/gather[dimension_numbers=GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0, 1), start_index_map=(0, 1)) slice_sizes=(1, 1) unique_indices=True indices_are_sorted=True mode=GatherScatterMode.PROMISE_IN_BOUNDS fill_value=None]\x00private\x00matmul\x00wrapped\x00apply_kernel\x00jax.result_info\x00\x00main\x00public\x00{"custom_call_config": {"body": "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EwMDsgIDAxMDA7oCAwMTAwPCAgMDEwMDygIDAxMDA9ICAwMTAwPaAgMDEwMD4gIDAxMDA+oCAwMTAwPyAgMDEwMN+gIDAREGDwMbAwURBg8DHQMHEQYPAx8DCQcHAwEDAQeNiYkHBwMBAwEHjYmFBwcDAQMBB42DiQcHAwEDAQeNg4UHBwMBAwEHjYGJBwcDAQMBB42BhQcHAwEDAQeNf4kHBwMBAwEHjX+FBwcDAQMBB419iQcHAwEDAQeNfYUHBwMBAwEHjXuJBwcDAQMBB417hQcHAwEDAQeNeYkHBwMBAwEHjXmFBwcDAQMBB413iQcHAwEDAQeNd4UHBwMBAwEHjXWJBwcDAQMBB411hQcHAwEDAQeNc4kHBwMBAwEHjXOFBwcDAQMBB41xiQcHAwEDAQeNcYUHBwMBAwEHjW+JBwcDAQMBB41vhQcHAwEDAQeNbYkHBwMBAwEHjW2FBwcDAQMBB41riQcHAwEDAQeNa4UHBwMBAwEHjWmJBwcDAQMBB41phQcHAwEDAQeNZ4kHBwMBAwEHjWeFBwcDAQMBB42FiQcHAwEDAQeNhYUHBwMBAwEHjWWJBwcDAQMBB41lhQcHAwEDAQeNY4kHBwMBAwEHjWOFBwcDAQMBB41hiQcHAwEDAQeNYYUHBwMBAwEHjV+JBwcDAQMBB41fhQcHAwEDAQeNXYkHBwMBAwEHjV2FBwcDAQMBB41biQcHAwEDAQeNW4UHBwMBAwEHjVmJBwcDAQMBB41ZhQcHAwEDAQeNV4kHBwMBAwEHjVeFBwcDAQMBB41ViQcHAwEDAQeNVYUHBwMBAwEHjVOJBwcDAQMBB41ThQcHAwEDAQeNUYkHBwMBAwEHjVGFBwcDAQMBB41PiQcHAwEDAQeNT4UHBwMBAwEHjU2JBwcDAQMBB41NhQcHAwEDAQeNS4kHBwMBAwEHjUuFBwcDAQMBB41JiQcHAwEDAQeNSYUHBwUBAwEHj4mJBwcFAQMBB4+JhQcHBQEDAQePg4kHBwUBAwEHj4OFBwcFAQMBB4+BiQcHBQEDAQePgYUHBwUBAwEHj3+JBwcFAQMBB49/hQcHBQEDAQePfYkHBwUBAwEHj32FBwcFAQMBB497iQcHBQEDAQePe4UHBwUBAwEHj3mJBwcFAQMBB495hQcHBQEDAQePd4kHBwUBAwEHj3eFBwcFAQMBB491iQcHBQEDAQePdYUHBwUBAwEHj3OJBwcFAQMBB49zhQcHBQEDAQePcYkHBwUBAwEHj3GFBwcFAQMBB49viQcHBQEDAQePb4UHBwUBAwEHj22JBwcFAQMBB49thQcHBQEDAQePa4kHBwUBAwEHj2uFBwcFAQMBB49piQcHBQEDAQePaYUHBwUBAwEHj2eJBwcFAQMBB49nhQcHBQEDAQePhYkHBwUBAwEHj4WFBwcFAQMBB49liQcHBQEDAQePZYUHBwUBAwEHj2OJBwcFAQMBB49jhQcHBQEDAQePYYkHBwUBAwEHj2GFBwcFAQMBB49fiQcHBQEDAQePX4UHBwUBAwEHj12JBwcFAQMBB49dhQcHBQEDAQePW4kHBwUBAwEHj1uFBwcFAQMBB49ZiQcHBQEDAQePWYUHBwUBAwEHj1eJBwcFAQMBB49XhQcHBQEDAQePVYkHBwUBAwEHj1WFBwcFAQMBB49TiQcHBQEDAQePU4UHBwUBAwEHj1GJBwcFAQMBB49RhQcHBQEDAQePT4kHBwUBAwEHj0+FBwcFAQMBB49NiQcHBQEDAQePTYUHBwUBAwEHj0uJBwcFAQMBB49LhQcHBQEDAQePSYkHBwUBAwEHj0mFBwcDAQMBB42JhwcHAwEDAQeNiUcHBwMBAwEHjYOHBwcDAQMBB42DRwcHAwEDAQeNgYcHBwMBAwEHjYFHBwcDAQMBB41/hwcHAwEDAQeNf0cHBwMBAwEHjX2HBwcDAQMBB419RwcHAwEDAQeNe4cHBwMBAwEHjXtHBwcDAQMBB415hwcHAwEDAQeNeUcHBwMBAwEHjXeHBwcDAQMBB413RwcHAwEDAQeNdYcHBwMBAwEHjXVHBwcDAQMBB41zhwcHAwEDAQeNc0cHBwMBAwEHjXGHBwcDAQMBB41xRwcHAwEDAQeNb4cHBwMBAwEHjW9HBwcDAQMBB41thwcHAwEDAQeNbUcHBwMBAwEHjWuHBwcDAQMBB41rRwcHAwEDAQeNaYcHBwMBAwEHjWlHBwcDAQMBB41nhwcHAwEDAQeNZ0cHBwMBAwEHjYWHBwcDAQMBB42FRwcHAwEDAQeNZYcHBwMBAwEHjWVHBwcDAQMBB41jhwcHAwEDAQeNY0cHBwMBAwEHjWGHBwcDAQMBB41hRwcHAwEDAQeNX4cHBwMBAwEHjV9HBwcDAQMBB41dhwcHAwEDAQeNXUcHBwMBAwEHjVuHBwcDAQMBB41bRwcHAwEDAQeNWYcHBwMBAwEHjVlHBwcDAQMBB41XhwcHAwEDAQeNV0cHBwMBAwEHjVWHBwcDAQMBB41VRwcHAwEDAQeNU4cHBwMBAwEHjVNHBwcDAQMBB41RhwcHAwEDAQeNUUcHBwMBAwEHjU+HBwcDAQMBB41PRwcHAwEDAQeNTYcHBwMBAwEHjU1HBwcDAQMBB41LhwcHAwEDAQeNS0cHBwMBAwEHjUmHBwcDAQMBB41JRwcHBQEDAQePh4kHBwUBAwEHj4eFBwcFAQMBB49FiQcHBQEDAQePRYUHBwUBAwEHj0OJBwcFAQMBB49DhQcHBQEDAQePQYkHBwUBAwEHj0GFBwcFAQMBB48/iQcHBQEDAQePP4UHBwUBAwEHjz2JBwcFAQMBB489hQcHBQEDAQePO4kHBwUBAwEHjzuFBwcFAQMBB485iQcHBQEDAQePOYUHBwUBAwEHjzeJBwcFAQMBB483hQcHBQEDAQePNYkHBwUBAwEHjzWFBwcFAQMBB48ziQcHBQEDAQePM4UHBwUBAwEHjzGJBwcFAQMBB48xhQcHBQEDAQePL4kHBwUBAwEHjy+FBwcFAQMBB48tiQcHBQEDAQePLYUHBwUBAwEHjyuJBwcFAQMBB48rhQcHBQEDAQePKYkHBwUBAwEHjymFBwcFAQMBB49HiQcHBQEDAQePR4UHBwUBAwEHjyeJBwcFAQMBB48nhQcHBQEDAQePJYkHBwUBAwEHjyWFBwcFAQMBB48jiQcHBQEDAQePI4UHBwUBAwEHjyGJBwcFAQMBB48hhQcHBQEDAQePH4kHBwUBAwEHjx+FBwcFAQMBB48diQcHBQEDAQePHYUHBwUBAwEHjxuJBwcFAQMBB48bhQcHBQEDAQePGYkHBwUBAwEHjxmFBwcFAQMBB48XiQcHBQEDAQePF4UHBwUBAwEHjxWJBwcFAQMBB48VhQcHBQEDAQePE4kHBwUBAwEHjxOFBwcFAQMBB48RiQcHBQEDAQePEYUHBwUBAwEHjw+JBwcFAQMBB48PhQcHBQEDAQePDYkHBwUBAwEHjw2FBwcFAQMBB48LiQcHBQEDAQePC4ULBw0RAwVBJgMuAzYDPgNGA04DVgNeA2YDbgN2A34DhgOOA5YDngOmA64DtgO+A8YDzgPWA94D5gPuA/YD/gMGBA4EFgQeBAsHDREDBUEqAzIDOgNCA0oDUgNaA2IDagNyA3oDggOKA5IDmgOiA6oDsgO6A8IDygPSA9oD4gPqA/ID+gMCBAoEEgQaBCIECwcNEQMLISYELgQ2BD4ERgROBFYEXgRmBG4EdgR+BIYEjgSWBJ4ECwcNEQMFQYuLi4uLi4uLi4uLi4uLi4uLi4uLi4uLi4uLi4uLi4uLDQcNEwMFByYFLgUyBQ8HDRVBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEDNgULBw0RAwshpgSuBLYEvgTGBM4E1gTeBOYE7gT2BP4EBgUOBRYFHgULBw0RAwVBOgU+BUIFRgVKBU4FUgVWBVoFXgViBWYFagVuBXIFdgV6BX4FggWGBYoFjgWSBZYFmgWeBaIFpgWqBa4FsgW2BQ0HDRMDBQcqBboFvgUPBw0VQQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA8IFCwcNEQMLISoEMgQ6BEIESgRSBFoEYgRqBHIEegSCBIoEkgSaBKIECwcNEQMFQYuLi4uLi4uLi4uLi4uLi4uLi4uLi4uLi4uLi4uLi4uLDQcNEwMFByYFRgZKBg8HDRVBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEDTgYLBw0RAwshqgSyBLoEwgTKBNIE2gTiBOoE8gT6BAIFCgUSBRoFIgULBw0RAwVBUgZWBloGXgZiBmYGagZuBnIGdgZ6Bn4GggaGBooGjgaSBpYGmgaeBqIGpgaqBq4Gsga2BroGvgbCBsYGygbOBg0HDRMDBQcqBdIG1gYPBw0VQQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA9oGCwcNEQMFQZOXm5+jp6uvs7e7v8PHy8/T19vf4+fr7/P3+/8GAg4CFgIeAgsHDREDBUGVmZ2hpamtsbW5vcHFyc3R1dnd4eXp7fH1+f0CAgoCEgIaAiICCwcNEQMLISYCLgI2Aj4CRgJOAlYCXgJmAm4CdgJ+AoYCjgKWAp4CCwcNEQMFQcYFygXOBdIF1gXaBd4F4gXmBeoF7gXyBfYF+gX+BQIGBgYKBg4GEgYWBhoGHgYiBiYGKgYuBjIGNgY6Bj4GQgYNBw0TAwUHXgdmB2oHDwcNFUEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQNuBwsHDREDCyGmAq4CtgK+AsYCzgLWAt4C5gLuAvYC/gIGAw4DFgMeAwsHDREDBUFyB3YHegd+B4IHhgeKB44HkgeWB5oHngeiB6YHqgeuB7IHtge6B74HwgfGB8oHzgfSB9YH2gfeB+IH5gfqB+4HDQcNEwMFB2IH8gf2Bw8HDRVBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQED+gcLBw0RAwshKgIyAjoCQgJKAlICWgJiAmoCcgJ6AoICigKSApoCogILBw0RAwVB3gbiBuYG6gbuBvIG9gb6Bv4GAgcGBwoHDgcSBxYHGgceByIHJgcqBy4HMgc2BzoHPgdCB0YHSgdOB1IHVgdaBw0HDRMDBQdeB34IgggPBw0VQQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA4YICwcNEQMLIaoCsgK6AsICygLSAtoC4gLqAvIC+gICAwoDEgMaAyIDCwcNEQMFQYoIjgiSCJYImgieCKIIpgiqCK4Isgi2CLoIvgjCCMYIygjOCNII1gjaCN4I4gjmCOoI7gjyCPYI+gj+CAIJBgkNBw0TAwUHYgcKCQ4JDwcNFUEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQMSCQkFCwkJ/geRiYkJBQsJCRYJkYmFCQULCQkCCJGDiQkFCwkJGgmRg4UJBQsJCQYIkYGJCQULCQkeCZGBhQkFCwkJCgiRf4kJBQsJCSIJkX+FCQULCQkOCJF9iQkFCwkJJgmRfYUJBQsJCRIIkXuJCQULCQkqCZF7hQkFCwkJFgiReYkJBQsJCS4JkXmFCQULCQkaCJF3iQkFCwkJMgmRd4UJBQsJCR4IkXWJCQULCQk2CZF1hQkFCwkJIgiRc4kJBQsJCToJkXOFCQULCQkmCJFxiQkFCwkJPgmRcYUJBQsJCSoIkW+JCQULCQlCCZFvhQkFCwkJLgiRbYkJBQsJCUYJkW2FCQULCQkyCJFriQkFCwkJSgmRa4UJBQsJCTYIkWmJCQULCQlOCZFphQkFCwkJOgiRZ4kJBQsJCVIJkWeFCQULCQk+CJGFiQkFCwkJVgmRhYUJBQsJCUIIkWWJCQULCQlaCZFlhQkFCwkJRgiRY4kJBQsJCV4JkWOFCQULCQlKCJFhiQkFCwkJYgmRYYUJBQsJCU4IkV+JCQULCQlmCZFfhQkFCwkJUgiRXYkJBQsJCWoJkV2FCQULCQlWCJFbiQkFCwkJbgmRW4UJBQsJCVoIkVmJCQULCQlyCZFZhQkFCwkJXgiRV4kJBQsJCXYJkVeFCQULCQliCJFViQkFCwkJegmRVYUJBQsJCWYIkVOJCQULCQl+CZFThQkFCwkJagiRUYkJBQsJCYIJkVGFCQULCQluCJFPiQkFCwkJhgmRT4UJBQsJCXIIkU2JCQULCQmKCZFNhQkFCwkJdgiRS4kJBQsJCY4JkUuFCQULCQl6CJFJiQkFCwkJkgmRSYUFAQ8eAwMRD2UHAwcLBQcPBw8TAw8vAwcFBA8FAQUDEQ9nBwMHCwUHDwcPEwMPLwMHBQQPBQUDAxEPaQcDBQcFBw8HDwUEDwUBAwYDAQUBAE4VTXYDKR0dF04C/gOB/gMdCyEjKR8bGRkZHSUTHRUNEykfDxsNCw8PDQkLEWJ1aWx0aW4AZnVuYwB0cHUAYXJpdGgAbW9kdWxlAHJldHVybgBsb2FkAHN0b3JlAHJvbGxfdmVjdG9ycwBtYXRtdWwAdW5yb2xsX3ZlY3RvcnMAZXJhc2VfbWVtcmVmX2xheW91dABjb25zdGFudAB2YWx1ZQBpbl9sYXlvdXQAZnVuY3Rpb25fdHlwZQBzeW1fbmFtZQB0cmFuc2Zvcm1faW5kaWNlcwB3aW5kb3dfYm91bmRzAHRyYW5zZm9ybV8wAHRyYW5zZm9ybV8xAHRyYW5zZm9ybV8yAHN1YmxhbmVfbWFzawBzdWJsYW5lX3N0cmlkZQBkaW1lbnNpb25fc2VtYW50aWNzAGl0ZXJhdGlvbl9ib3VuZHMAc2NhbGFyX3ByZWZldGNoAG1haW4Ad2luZG93X3BhcmFtcwAvbWFza2VkX2xvYWRbbWFza2VkPUZhbHNlIGNhY2hlX21vZGlmaWVyPSBldmljdGlvbl9wb2xpY3k9IGlzX3ZvbGF0aWxlPUZhbHNlIGFyZ3NfdHJlZT1QeVRyZWVEZWYoKEN1c3RvbU5vZGUoTkRJbmRleGVyWyhbXSwgUHlUcmVlRGVmKFtDdXN0b21Ob2RlKFNsaWNlWyhGYWxzZSwgMjU2KV0sIFsqXSksIEN1c3RvbU5vZGUoU2xpY2VbKFRydWUsIDAsIDI1NildLCBbXSldKSwgW1RydWUsIFRydWVdLCAoNTEyLCAyNTYpLCAoKSldLCBbKl0pLCkpXQB0aGlyZF9wYXJ0eS9weS9qYXhfdHJpdG9uL2dvb2dsZS9wYWxsYXNfdHB1L2JhY2tfY29tcGF0X3Rlc3QucHkAL21hc2tlZF9sb2FkW21hc2tlZD1GYWxzZSBjYWNoZV9tb2RpZmllcj0gZXZpY3Rpb25fcG9saWN5PSBpc192b2xhdGlsZT1GYWxzZSBhcmdzX3RyZWU9UHlUcmVlRGVmKChDdXN0b21Ob2RlKE5ESW5kZXhlclsoW10sIFB5VHJlZURlZihbQ3VzdG9tTm9kZShTbGljZVsoVHJ1ZSwgMCwgMjU2KV0sIFtdKSwgQ3VzdG9tTm9kZShTbGljZVsoRmFsc2UsIDI1NildLCBbKl0pXSksIFtUcnVlLCBUcnVlXSwgKDI1NiwgNTEyKSwgKCkpXSwgWypdKSwpKV0AL2RvdF9nZW5lcmFsW2RpbWVuc2lvbl9udW1iZXJzPSgoKDEsKSwgKDAsKSksICgoKSwgKCkpKSBwcmVjaXNpb249KDxQcmVjaXNpb24uREVGQVVMVDogMD4sIDxQcmVjaXNpb24uREVGQVVMVDogMD4pIHByZWZlcnJlZF9lbGVtZW50X3R5cGU9ZmxvYXQzMl0Ab3V0X2xheW91dAB0cmFuc3Bvc2VfbGhzAHRyYW5zcG9zZV9yaHMAb3BlcmFuZFNlZ21lbnRTaXplcwAvbWFza2VkX3N3YXBbbWFza2VkPUZhbHNlIGV2aWN0aW9uX3BvbGljeT0gYXJnc190cmVlPVB5VHJlZURlZigoQ3VzdG9tTm9kZShOREluZGV4ZXJbKFtdLCBQeVRyZWVEZWYoW0N1c3RvbU5vZGUoU2xpY2VbKFRydWUsIDAsIDI1NildLCBbXSksIEN1c3RvbU5vZGUoU2xpY2VbKFRydWUsIDAsIDI1NildLCBbXSldKSwgW1RydWUsIFRydWVdLCAoMjU2LCAyNTYpLCAoKSldLCBbXSksKSldAA=="}}\x00tpu_custom_call\x00func\x00',
    xla_call_module_version=7,
)  # End paste
