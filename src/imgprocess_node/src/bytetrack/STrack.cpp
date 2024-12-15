#include "STrack.h"

/**
 * @brief 轨迹类的构造函数
 * @param tlwh_是传入的观测数据即检测框的左上点和长宽,score为检测得分即置信度
 * @param 初始状态is_activated为false即未激活
*/
STrack::STrack( std::vector<float> tlwh_, float score) {
	_tlwh.resize(4);
	_tlwh.assign(tlwh_.begin(), tlwh_.end());

	is_activated = false;
	track_id = 0;
	state = TrackState::New;
	
	tlwh.resize(4);
	tlbr.resize(4);

	static_tlwh();
	static_tlbr();
	frame_id = 0;
	tracklet_len = 0;
	this->score = score;
	start_frame = 0;
}

STrack::~STrack() {

}

/**
 * @brief 激活轨迹,一般只有第一帧会将轨迹的is_activated置一,对中途的新轨迹很保守
 * @param 输入为卡尔曼滤波器和跟踪帧数
*/
void STrack::activate(byte_kalman::ByteKalmanFilter &kalman_filter, int frame_id) {
	this->kalman_filter = kalman_filter;
	this->track_id = this->next_id();

	std::vector<float> _tlwh_tmp(4);
	_tlwh_tmp[0] = this->_tlwh[0];
	_tlwh_tmp[1] = this->_tlwh[1];
	_tlwh_tmp[2] = this->_tlwh[2];
	_tlwh_tmp[3] = this->_tlwh[3];
	std::vector<float> xyah = tlwh_to_xyah(_tlwh_tmp);
	DETECTBOX xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];
	auto mc = this->kalman_filter.initiate(xyah_box);
	this->mean = mc.first;
	this->covariance = mc.second;

	static_tlwh();
	static_tlbr();

	this->tracklet_len = 0;
	this->state = TrackState::Tracked;
	if (frame_id == 1) {
		this->is_activated = true;
	}
	// this->is_activated = true;
	this->frame_id = frame_id;
	this->start_frame = frame_id;
}

/**
 * @brief 重新激活轨迹
 * @param 输入为轨迹,帧数,新帧数false
 * @bug 进行了一次卡尔曼滤波并更新了状态矩阵和协方差矩阵
 * 并且把state标记为Tracked,实际是第一次匹配中为跟上的轨迹才会进这个函数
 * 之后又筛选Tracked的轨迹,感觉很多余,被那进来的本来就是已经全部被赋Tracked了吧
*/
void STrack::re_activate(STrack &new_track, int frame_id, bool new_id) {
	std::vector<float> xyah = tlwh_to_xyah(new_track.tlwh);
	DETECTBOX xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];
	auto mc = this->kalman_filter.update(this->mean, this->covariance, xyah_box);
	this->mean = mc.first;
	this->covariance = mc.second;

	static_tlwh();
	static_tlbr();

	this->tracklet_len = 0;
	this->state = TrackState::Tracked;
	this->is_activated = true;
	this->frame_id = frame_id;
	this->score = new_track.score;
	if (new_id) {
		this->track_id = next_id();
	}
}

/**
 * @brief 对跟踪上的轨迹进行卡尔曼滤波更新状态矩阵和协方差矩阵
 * @param 输入为轨迹,帧数
 * @bug 进行了一次卡尔曼滤波并更新了状态矩阵和协方差矩阵
*/
void STrack::update(STrack &new_track, int frame_id) {
	this->frame_id = frame_id;
	this->tracklet_len++;

	std::vector<float> xyah = tlwh_to_xyah(new_track.tlwh);
	DETECTBOX xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];

	auto mc = this->kalman_filter.update(this->mean, this->covariance, xyah_box);
	this->mean = mc.first;
	this->covariance = mc.second;

	static_tlwh();
	static_tlbr();

	this->state = TrackState::Tracked;
	this->is_activated = true;

	this->score = new_track.score;
}

/**
 * @brief 初次创建轨迹时,将_tlwh值赋给tlwh
 * 之后通过mean值还原tlwh
*/
void STrack::static_tlwh() {
	if (this->state == TrackState::New) {
		tlwh[0] = _tlwh[0];
		tlwh[1] = _tlwh[1];
		tlwh[2] = _tlwh[2];
		tlwh[3] = _tlwh[3];
		return;
	}

	tlwh[0] = mean[0];
	tlwh[1] = mean[1];
	tlwh[2] = mean[2];
	tlwh[3] = mean[3];

	tlwh[2] *= tlwh[3];
	tlwh[0] -= tlwh[2] / 2;
	tlwh[1] -= tlwh[3] / 2;
}

/**
 * @brief 通过tlwh得到tlbr
*/
void STrack::static_tlbr() {
	tlbr.clear();
	tlbr.assign(tlwh.begin(), tlwh.end());
	tlbr[2] += tlbr[0];
	tlbr[3] += tlbr[1];
}

std::vector<float> STrack::tlwh_to_xyah( std::vector<float> tlwh_tmp) {
	 std::vector<float> tlwh_output = tlwh_tmp;
	tlwh_output[0] += tlwh_output[2] / 2;
	tlwh_output[1] += tlwh_output[3] / 2;
	tlwh_output[2] /= tlwh_output[3];
	return tlwh_output;
}

std::vector<float> STrack::to_xyah() {
	return tlwh_to_xyah(tlwh);
}

std::vector<float> STrack::tlbr_to_tlwh( std::vector<float> &tlbr) {
	tlbr[2] -= tlbr[0];
	tlbr[3] -= tlbr[1];
	return tlbr;
}

void STrack::mark_lost() {
	state = TrackState::Lost;
}

void STrack::mark_removed() {
	state = TrackState::Removed;
}

/**
 * 获得新的帧数
*/
int STrack::next_id() {
	static int _count = 0;
	_count++;
	return _count;
}

int STrack::end_frame() {
	return this->frame_id;
}

/**
 * @brief 对初步跟踪轨迹进行卡尔曼预测的接口
 * @return 对输入轨迹的mean和covariance作出改变
*/
void STrack::multi_predict(std::vector<STrack*> &stracks, byte_kalman::ByteKalmanFilter &kalman_filter) {
	for (int i = 0; i < stracks.size(); i++) {
		if (stracks[i]->state != TrackState::Tracked) {
			stracks[i]->mean[7] = 0;
		}
		kalman_filter.predict(stracks[i]->mean, stracks[i]->covariance);
	}
}