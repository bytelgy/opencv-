# 实战--银行卡号识别
# 根据长宽比例取出银行卡的数字组合
# 模板匹配，将银行卡的数字与模板中的每一个区域进行匹配
# 外轮廓检测+外接矩形，取出模板中的每个值
# 银行卡数字大小resize和模板数字一样大

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 指定图片所在文件夹
filepath = 'D:\\PC\\images'
# 图像显示函数
def cv_show(name,img):
    cv2.imshow(name,img) # (自定义图像名,图像变量)
    cv2.waitKey(0) # 图像窗口不会自动关闭
    cv2.destroyAllWindows()  # 手动关闭窗口

#（1）模板处理
# 读取模板图像
reference = cv2.imread(filepath+'\\reference.png')
cv_show('reference',reference) #绘图
# 转换灰度图，颜色改变函数
ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
cv_show('gray',ref)
# 二值化处理，图像阈值函数，像素值超过127变成0，否则变成255
ret,thresh = cv2.threshold(ref,127,255,cv2.THRESH_BINARY_INV)
cv_show('threshold',thresh) # 返回值ret是阈值，thresh是二值化图像
# 轮廓检测，轮廓检测函数。第2个参数检测最外层轮廓，第3个参数保留轮廓终点坐标
image,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 返回轮廓信息和轮廓层数
# 绘制轮廓
draw = reference.copy()  # 复制一份原图像作为画板，不能在原图上画，不然原图会改变
res = cv2.drawContours(draw, contours, -1, (0,0,255), 2)  #在画板上画出所有轮廓信息"-1"，红色线宽为2
cv_show('res',res)
print(np.array(contours).shape)  # 显示有多少个轮廓

#（2）模板排序
# 由于现在的轮廓不一定contours[0]对应的就是数字0，我们需要对它排序
# 求每一个轮廓的外接矩形，根据返回的左上坐标点，就能判断出轮廓的位置，再排序
# boxing中存放每次计算轮廓外接矩形得到的x、y、w、h，shape=(10,4)。cnt存放每一个轮廓
boxing = [np.array(cv2.boundingRect(cnt)) for cnt in contours]
contours = np.array(contours)
# 都变成数组类型，下面冒泡排序能相互交换值，元组类型只读不能写
# 把x坐标最小轮廓的排在第一个
for i in range(9):  #冒泡排序
    for j in range(i+1,10):
        if boxing[i][0]>boxing[j][0]:  #把x坐标大的值放到后面
        # 给boxing中的值换位置
            boxing[i],boxing[j] = boxing[j],boxing[i]
        # 给轮廓信息换位置
            contours[i],contours[j] = contours[j],contours[i]

#（3）遍历每一个轮廓，给每一个轮廓对应具体数字
# 定义一个字典，具体数字对应具体框
dic = {}
for i,cnt in enumerate(contours):  #返回轮廓下标和对应的轮廓值
    x,y,w,h = boxing[i]  # boxing中存放的是每个轮廓的信息
    roi = ref[y:y+h,x:x+w] # ref是灰度图像，ref中依次保存的是高和宽，即(y,x)
    # 将每个区域对应一个数字
    roi = cv2.resize(roi,(25,30))
    dic[i] = roi # 如果觉得roi区域有点小，可以重新resize一下
    cv_show('roi',roi)  # 单独显示
    # 组合在一起看一下
    plt.subplot(2,5,i+1)
    plt.imshow(roi,'gray'),plt.xticks([]),plt.yticks([]) #不显示xy轴刻度

# 2 银行卡图像预处理
#（1）读入银行卡数据，转换灰度图
img = [] # 存放5张银行卡数据
for i in range(5): # 使用循环读取，注意图片名一定要统一
    img_read = cv2.imread(filepath+f'\\card{i+1}.png',0) # 0表示灰度图
    img.append(img_read)
cv_show('card1',img[0])

#（2）形态学处理，使用礼帽操作，获取原图像中比周围亮的区域
img_treated = []  # 存放处理后的图片数据
for img_treat in img:
    # 定义一个卷积核，MORPH_RECT矩形，size为9*3
    rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,3))  #每一组数字的长宽比是9*3
    # 礼帽，突出更明亮的区域
    tophat = cv2.morphologyEx(img_treat,cv2.cv2.MORPH_TOPHAT,rectkernel)
    img_treated.append(tophat)
# 绘图看一下
cv_show('hat',img_treated[0])
    
#（3）边缘检测，canny边缘检测
img_cannyed = []  #存放处理后的数据
for img_canny in img_treated:
    img_canny = cv2.Canny(img_canny,80,200)  #自定义最小和最大阈值
    img_cannyed.append(img_canny)
# 绘图看一下
cv_show('canny',img_cannyed[0])

#（4）通过闭操作（先膨胀后腐蚀），将轮廓连在一起
img_closed = []  #存放闭操作之后的结果
for img_close in img_cannyed:
    img_close = cv2.morphologyEx(img_close, cv2.MORPH_CLOSE, rectkernel,iterations=3)  #使用第(3)步定义的9*3的卷积核
    img_closed.append(img_close)
# 绘图看一下
cv_show('closed',img_closed[0])

# #（5）二值化处理
# # THRESH_OUT会自动寻找合适的阈值，合适的双峰（两种主体），把阈值参数置为0
# img_threshed = []
# for img_thresh in img_closed:
#     ret,thresh = cv2.threshold(img_thresh,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     img_threshed.append(thresh)
# cv_show('threshed',img_threshed[0])

# #（6）闭操作，填补黑色空洞
# img_aclosed = []  #存放闭操作之后的结果
# for img_close in img_threshed:
#     img_close = cv2.morphologyEx(img_close, cv2.MORPH_CLOSE, rectkernel,iterations=2)  #使用第(3)步定义的9*3的卷积核
#     img_aclosed.append(img_close)
# # 绘图看一下
# cv_show('closed',img_aclosed[3])


#（5）计算轮廓
img_cnted = []
num = 0 # 原图像的索引号
for img_cnt in img_closed:
    # 轮廓检测，只绘制最外层轮廓
    image,contours,hierarchy = cv2.findContours(img_cnt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 返回轮廓信息和轮廓层数
    # 绘制轮廓
    draw = img[num].copy() ;num += 1 # 复制一份原图像作为画板，不能在原图上画，不然原图会改变
    res = cv2.drawContours(draw, contours, -1, (0,255,0), 2)  #在画板上画出所有轮廓信息"-1"，红色线宽为2
    cv_show('res',res)
    img_cnted.append(contours)


#（6）画出轮廓外接矩形
# img_rectangle = []
# num = 0 #指定原图下标
# for img_rec in img_cnted:
#     # boxing存放矩形要素,(x,y)是左上角坐标
#     x,y,w,h = cv2.boundingRect(img_rec)
#     # 绘制矩形框，在第num张原图上绘制矩形
#     rectangle = cv2.rectangle(img[num],(x,y),(x+w,y+h),(0,0,255),2)
#     num += 1 # 切换下一张图
#     img_rectangle.append(rectangle)
# cv_show('rectangle',img_rectangle[0])


#（6）筛选轮廓，根据长宽比
loc = [] # 存放img_cnted[0]图片排序后的轮廓要素
mess = [] # 每img_cnted[0]图片排序后的轮廓信息
for (i,c) in enumerate(img_cnted[0]): #返回下标和对应值
    # 每一个轮廓的外接矩形要素
    (x,y,w,h) = cv2.boundingRect(c)
    ar = w/float(h)  # 计算长宽比
    # 选择合适的长宽比
    if ar>2.5 and ar<4:
        if (w>90 and w<100) and (h>25 and h<35):
            # 符合要求的留下
            loc.append((x,y,w,h))
            mess.append(c)
            # 将符合的轮廓从左到右排序
            # 把x坐标最小轮廓的排在第一个
for i in range(len(loc)-1):  #冒泡排序
    for j in range(i+1,len(loc)):
        if loc[i][0]>loc[j][0]:  #把x坐标大的值放到后面
            # 交换轮廓要素信息
            loc[i],loc[j] = loc[j],loc[i]
            # 交换对应的轮廓信息
            mess[i],mess[j] = mess[j],mess[i]

n = 1
output = []  # 保存最终结果
#（7）根据轮廓提取每一组数字组合
for (i, (x,y,w,h)) in enumerate(loc): # loc中存放的是每一个组合的xywh
    groupoutput = []  #存放取出来的数字组合
    group = img[0][y-5:y+h+5,x-5:x+w+5]  # 每个组合的坐标范围是[x:x+w][y:y+h]，加减5是为了给周围留点空间
    cv_show('group',group)
    # # 组合在一起看一下
    # plt.subplot(1,4,i+1)
    # plt.imshow(group,'gray'),plt.xticks([]),plt.yticks([]) #不显示xy轴刻度
    
    
#（8）每次取出的轮廓预处理，二值化
    # THRESH_OUT会自动寻找合适的阈值，合适的双峰（两种主体），把阈值参数置为0
    ret,group = cv2.threshold(group,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU) #二值化处理
    cv_show('group',group) #ret是返回的阈值，group是返回的二值化后的图像
    # # 组合在一起看一下
    # plt.subplot(1,4,i+1)
    # plt.imshow(group,'gray'),plt.xticks([]),plt.yticks([]) #不显示xy轴刻度
    

#（9）每个数字的小轮廓检测，只检测最外层轮廓RETR_EXTERNAL，返回轮廓各个终点的坐标CHAIN_APPROX_SIMPLE
    image,contours,hierarchy = cv2.findContours(group, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 返回轮廓信息和轮廓层数
    # 对轮廓排序，boxing中存放每次计算轮廓外接矩形得到的x、y、w、h
    boxing = [np.array(cv2.boundingRect(cnt)) for cnt in contours]
    contours = np.array(contours)   # 都变成数组类型，下面冒泡排序能相互交换值，元组类型只读不能写
    # 把x坐标最小轮廓的排在第一个
    for i in range(3):  #冒泡排序
        for j in range(i+1,4):
            if boxing[i][0]>boxing[j][0]:  #把x坐标大的值放到后面
                # 给boxing中的值换位置
                boxing[i],boxing[j] = boxing[j],boxing[i]
                # 给轮廓信息换位置
                contours[i],contours[j] = contours[j],contours[i]
    
#（10）给排序后的轮廓分别计算每一个数字组合中的每一个数字
    for c in contours: # c代表每一个数字小轮廓
        (gx,gy,gw,gh) = cv2.boundingRect(c) # 计算每一个数字小轮廓的x,y,w,h
        roi = group[gy:gy+gh,gx:gx+gw]  # 在数字组合中扣除每一个数字区域
        roi = cv2.resize(roi,(25,30))  # 大小和最开始resize的模板大小一样
        cv_show('roi',roi)  # 扣出了所有的数字
        # # 组合在一起看一下
        # plt.subplot(1,17,n+1)
        # plt.imshow(roi,'gray'),plt.xticks([]),plt.yticks([]) #不显示xy轴刻度
        # n += 1
        
#（11）开始模板匹配，对每一个roi进行匹配
        score = []  # 定义模板匹配度得分变量
        # 从模板中逐一取出数字和刚取出来的roi比较
        for (dic_key, dic_value) in dic.items(): # items()函数从我们最开始定义的模板字典中取出索引和值
            # 模板匹配，计算归一化相关系数cv2.TM_CCOEFF_NORMED，计算结果越接近1，越相关
            res = cv2.matchTemplate(roi,dic_value,cv2.TM_CCOEFF_NORMED)
            # 返回最值及最值位置，在这里我们需要的是最小值的得分，不同的匹配度计算方法的选择不同
            min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)
            score.append(max_val)
        # 当roi与模板中的10个数比较完以后，score中保存的是roi和每一个数字匹配后的结果
        # score中值最小的位置，就是roi对应的数字
        score = np.abs(score) # 有负数出现，统一成正数，看相关系数
        best_index = np.argmax(score)  # score最大值的下标，匹配度最高
        best_value = str(best_index)  # 下标就是对应的数字，在字典中，key是0对应的是值为0的图片
        groupoutput.append(best_value)  # 将对应的数字保存
    # 打印识别结果
    print(groupoutput)
    
#（12）把我们识别处理的数字在原图上画出来，指定矩形框的左上角坐标和右下角坐标
    cv2.rectangle(img[0],(x-5,y-5),(x+w+5,y+h+5),(0,0,255),1)
    # 在矩形框上绘图
    cv2.putText(img[0], ''.join(groupoutput), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)
    # 将得到的数字结果保存在一起
    output.append(groupoutput)

#（13）循环结束，展示结果
cv_show('img',img[0])
print('数字为：',output)
