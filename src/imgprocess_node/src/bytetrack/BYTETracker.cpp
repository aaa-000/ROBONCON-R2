#include "BYTETracker.h"
#include <fstream>

BYTETracker::BYTETracker(int frame_rate, int track_buffer) {
	track_thresh = 0.5; // 评测检测高低分的检测置信阈值
	high_thresh = 0.6;	// 中途创建新物体的检测置信阈值
	match_thresh = 0.8; // 过滤iou过小的轨迹和检测

	frame_id = 0;
	max_time_lost = int(frame_rate / 30.0 * track_buffer);
}

BYTETracker::~BYTETracker() {

}

/**
 * @brief 跟踪器的更新函数
 * @param 输入为检测的结果容器
 * @return 返回当前帧跟踪到的轨迹
*/
std::vector<STrack> BYTETracker::update(const std::vector<det::Object>& objects) {
	/*********第一步,对检测和追踪轨迹进行分类,前者分为高低分,后者分为是否激活*********/
	this->frame_id++;
	std::vector<STrack> activated_stracks;	// 当前帧匹配到追踪轨迹
	std::vector<STrack> refind_stracks;		// 当前帧匹配到失追轨迹
	std::vector<STrack> removed_stracks;	// 保存当前帧
	std::vector<STrack> lost_stracks;		// 保存当前帧没有匹配目标的轨迹,即上一帧还在持续追踪但是这一帧两次匹配不到的轨迹
	std::vector<STrack> detections;			// 高分检测
	std::vector<STrack> detections_low;		// 低分检测

	std::vector<STrack> detections_cp;
	std::vector<STrack> tracked_stracks_swap;
	std::vector<STrack> resa, resb;
	std::vector<STrack> output_stracks;

	std::vector<STrack*> unconfirmed;
	std::vector<STrack*> tracked_stracks;
	std::vector<STrack*> strack_pool;
	std::vector<STrack*> r_tracked_stracks;

	/**
	 * 遍历所有检测,将objects转换成x1,y1,x2,y2,score的格式,并以此创建轨迹
	 * 并根据置信度的大小分为高低分检测
	*/
	if (objects.size() > 0) {
		for (int i = 0; i < objects.size(); i++) {
			std::vector<float> tlbr_;
			tlbr_.resize(4);
            tlbr_[0] = objects[i].rect.x;
            tlbr_[1] = objects[i].rect.y;
            tlbr_[2] = objects[i].rect.x + objects[i].rect.width;
            tlbr_[3] = objects[i].rect.y + objects[i].rect.height;

            float score = objects[i].prob;

			STrack strack(STrack::tlbr_to_tlwh(tlbr_), score);
			if (score >= track_thresh) {
				detections.push_back(strack);
			} else {
				detections_low.push_back(strack);
			}
		}
	}

	// 将轨迹根据是否激活分到unconfirmed和tracked_stracks里面
	for (int i = 0; i < this->tracked_stracks.size(); i++) {
		if (!this->tracked_stracks[i].is_activated) {
			unconfirmed.push_back(&this->tracked_stracks[i]);
		} else {
			tracked_stracks.push_back(&this->tracked_stracks[i]);
		}
	}

	/*********第二步,对轨迹进行第一次追踪,针对激活状态的轨迹的高分匹配*********/
	strack_pool = joint_stracks(tracked_stracks, this->lost_stracks); // 将已追踪轨迹和失追轨迹合并为初步跟踪
	STrack::multi_predict(strack_pool, this->kalman_filter); // 对轨迹进行卡尔曼预测

	// 计算轨迹和检测之间的iou
	std::vector< std::vector<float> > dists;
	int dist_size = 0, dist_size_size = 0;
	dists = iou_distance(strack_pool, detections, dist_size, dist_size_size);

	// 使用匈牙利算法计算最小损失的匹配
	std::vector< std::vector<int> > matches;
	std::vector<int> u_track, u_detection;
	linear_assignment(dists, dist_size, dist_size_size, match_thresh, matches, u_track, u_detection);

	/**
	 * 遍历匹配对matches,如果轨迹被匹配上了(state==Tracked)，调用update方法，并加入到activated_stracks
	 * 否则调用re_activate，并加入refind_stracks
	*/
	for (int i = 0; i < matches.size(); i++) {
		STrack *track = strack_pool[matches[i][0]];
		STrack *det = &detections[matches[i][1]];
		if (track->state == TrackState::Tracked) {
			track->update(*det, this->frame_id);
			activated_stracks.push_back(*track);
		} else {
			track->re_activate(*det, this->frame_id, false);
			refind_stracks.push_back(*track);
		}
	}

	/*********第三步,对轨迹进行第二次追踪,针对前轮未被匹配的激活状态的轨迹的低分匹配*********/
	/**
	 * 将未被匹配的检测存到detections_cp里
	 * 将低分检测detections_low放到detections继续这轮匹配
	 * 将未被匹配的轨迹中激活状态的轨迹存到r_tracked_stracks里继续本轮匹配
	*/ 
	for (int i = 0; i < u_detection.size(); i++) {
		detections_cp.push_back(detections[u_detection[i]]);
	}
	detections.clear();
	detections.assign(detections_low.begin(), detections_low.end());
	for (int i = 0; i < u_track.size(); i++) {
		if (strack_pool[u_track[i]]->state == TrackState::Tracked) {
			r_tracked_stracks.push_back(strack_pool[u_track[i]]);
		}
	}

	dists.clear();
	dists = iou_distance(r_tracked_stracks, detections, dist_size, dist_size_size);
	matches.clear();
	u_track.clear();
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, 0.5, matches, u_track, u_detection);

	for (int i = 0; i < matches.size(); i++) {
		STrack *track = r_tracked_stracks[matches[i][0]];
		STrack *det = &detections[matches[i][1]];
		if (track->state == TrackState::Tracked) {
			track->update(*det, this->frame_id);
			activated_stracks.push_back(*track);
		} else {
			track->re_activate(*det, this->frame_id, false);
			refind_stracks.push_back(*track);
		}
	}

	// 遍历第二次也没匹配到的轨迹,改为Lost状态,并加入lost_stracks
	for (int i = 0; i < u_track.size(); i++) {
		STrack *track = r_tracked_stracks[u_track[i]];
		if (track->state != TrackState::Lost) {
			track->mark_lost();
			lost_stracks.push_back(*track);
		}
	}

	/**
	 * 尝试匹配中途第一次出现的轨迹
	 * 当前帧的目标框会优先和长期存在的轨迹（包括持续追踪的和断追的轨迹）匹配，再和只出现过一次的目标框匹配
	*/
	detections.clear();
	detections.assign(detections_cp.begin(), detections_cp.end());

	dists.clear();
	dists = iou_distance(unconfirmed, detections, dist_size, dist_size_size);

	matches.clear();
	std::vector<int> u_unconfirmed;
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, 0.7, matches, u_unconfirmed, u_detection);

	for (int i = 0; i < matches.size(); i++) {
		unconfirmed[matches[i][0]]->update(detections[matches[i][1]], this->frame_id);
		activated_stracks.push_back(*unconfirmed[matches[i][0]]);
	}
	// 第一次出现的轨迹如果下一帧没有跟上直接删除
	for (int i = 0; i < u_unconfirmed.size(); i++) {
		STrack *track = unconfirmed[u_unconfirmed[i]];
		track->mark_removed();
		removed_stracks.push_back(*track);
	}

	/*********第四步,前两次都没匹配上的高分检测,认为是新的物体,考虑赋予新的ID,创建新的轨迹*********/
	for (int i = 0; i < u_detection.size(); i++) {
		STrack *track = &detections[u_detection[i]];
		if (track->score < this->high_thresh) {
			continue;
		}
		track->activate(this->kalman_filter, this->frame_id);
		activated_stracks.push_back(*track);
	}

	/*********第五步,对状态进行更新*********/
	for (int i = 0; i < this->lost_stracks.size(); i++) {
		if (this->frame_id - this->lost_stracks[i].end_frame() > this->max_time_lost) {
			this->lost_stracks[i].mark_removed();
			removed_stracks.push_back(this->lost_stracks[i]);
		}
	}
	
	for (int i = 0; i < this->tracked_stracks.size(); i++) {
		if (this->tracked_stracks[i].state == TrackState::Tracked) {
			tracked_stracks_swap.push_back(this->tracked_stracks[i]);
		}
	}
	this->tracked_stracks.clear();
	this->tracked_stracks.assign(tracked_stracks_swap.begin(), tracked_stracks_swap.end());

	this->tracked_stracks = joint_stracks(this->tracked_stracks, activated_stracks);
	this->tracked_stracks = joint_stracks(this->tracked_stracks, refind_stracks);

	//std::cout << activated_stracks.size() << std::endl;

	this->lost_stracks = sub_stracks(this->lost_stracks, this->tracked_stracks);
	for (int i = 0; i < lost_stracks.size(); i++) {
		this->lost_stracks.push_back(lost_stracks[i]);
	}

	this->lost_stracks = sub_stracks(this->lost_stracks, this->removed_stracks);
	for (int i = 0; i < removed_stracks.size(); i++) {
		this->removed_stracks.push_back(removed_stracks[i]);
	}
	
	remove_duplicate_stracks(resa, resb, this->tracked_stracks, this->lost_stracks);

	this->tracked_stracks.clear();
	this->tracked_stracks.assign(resa.begin(), resa.end());
	this->lost_stracks.clear();
	this->lost_stracks.assign(resb.begin(), resb.end());
	
	for (int i = 0; i < this->tracked_stracks.size(); i++) {
		if (this->tracked_stracks[i].is_activated) {
			output_stracks.push_back(this->tracked_stracks[i]);
		}
	}
	return output_stracks;
}
