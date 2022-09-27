
class MakeJson(object):
    def __init__(self, ann_info, aug_num, img_info, tool_name):
        self.ann_info = ann_info
        self.tool_name = tool_name
        self.aug_num = aug_num
        self.img_info = img_info

    def json_main(self):
        cat_list, area_list, box_list, seg_list = [], [], {}, {}
        for idx, ann in enumerate(self.ann_info):
            cat_list.append(ann['category_id'])
            json_tool = self.return_json()
            tmp_box, tmp_seg = json_tool(ann)
            area_list.append(tmp_box[2] * tmp_box[3])

            box_list[idx] = tmp_box
            seg_list[idx] = tmp_seg

        return cat_list, area_list, box_list, seg_list

    def return_json(self):
        json_tool_box = {"gamma": self.json_original, "cutout": self.json_original,
                         "mirroring": self.json_mirroring, "gaussian": self.json_original}

        return json_tool_box[self.tool_name]

    def json_original(self, ann):
        box_pt = ann['bbox']
        seg_pt = ann['segmentation']
        return box_pt, seg_pt

    def json_mirroring(self, ann):
        new_bbox_list, new_seg_list = [], []
        bbox = ann['bbox']
        x, y, width, height = bbox[0], bbox[1], bbox[2], bbox[3]

        if self.aug_num == -1:  # 상하좌우
            new_x = self.img_info[1] - (x + width)
            new_y = self.img_info[2] - (y + height)
            new_bbox_list = [new_x, new_y, width, height]

            for idx, seg in enumerate(ann['segmentation'][-1]):
                if idx % 2 == 0:  # x 좌표
                    new_seg_x = self.img_info[1] - seg
                    new_seg_list.append(new_seg_x)
                if idx % 2 == 1:  # y 좌표
                    new_seg_y = self.img_info[2] - seg
                    new_seg_list.append(new_seg_y)

        if self.aug_num == 0:  # 상하
            tmp = self.img_info[2] - y
            new_y = tmp - height
            new_bbox_list = [x, new_y, width, height]

            new_seg_list = []
            for idx, seg in enumerate(ann['segmentation'][-1]):
                if idx % 2 == 1:  # y 좌표
                    new_seg_y = self.img_info[2] - seg
                    new_seg_list.append(new_seg_y)
                else:
                    new_seg_list.append(seg)

        if self.aug_num == 1:  # 좌우
            tmp = self.img_info[1] - x
            new_x = tmp - width
            new_bbox_list = [new_x, y, width, height]

            new_seg_list = []
            for idx, seg in enumerate(ann['segmentation'][-1]):
                if idx % 2 == 0:  # x 좌표
                    new_seg_x = self.img_info[1] - seg
                    new_seg_list.append(new_seg_x)
                else:
                    new_seg_list.append(seg)

        return new_bbox_list, [new_seg_list]
