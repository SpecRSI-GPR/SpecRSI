// Trusted ns-3 packet-level dumbbell, compiled by cppyy against ns3 3.44.
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/traffic-control-module.h"
#include "ns3/flow-monitor-module.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <vector>

namespace specrsi_native_aqm {
using namespace ns3;
struct ControlState {
 Ptr<RedQueueDisc> queue;
 Ptr<FlowMonitor> monitor;
 double minimum,maximum,gain,target,interval,duration,bandwidth;
 uint32_t limit;
 uint64_t previousRx=0;
 std::vector<std::vector<double>> trace;
};
void Control(ControlState* s) {
 double packets=s->queue->GetNPackets();
 double delay=0; uint64_t received=0,bytes=0;
 for (const auto& entry:s->monitor->GetFlowStats()) {
  delay+=entry.second.delaySum.GetSeconds(); received+=entry.second.rxPackets; bytes+=entry.second.rxBytes;
 }
 delay=received ? delay/received : 0;
 double utilization=std::min(1.0,(bytes-s->previousRx)*8.0/(s->interval*s->bandwidth)); s->previousRx=bytes;
 double factor=std::clamp(1.0+s->gain*(s->target-delay)/s->target,0.25,2.0);
 double maximum=std::clamp(s->maximum*factor,2.0,double(s->limit-1));
 double minimum=std::clamp(s->minimum*factor,1.0,maximum-1);
 s->queue->SetTh(minimum,maximum);
 auto stats=s->queue->GetStats();
 s->trace.push_back({Simulator::Now().GetSeconds(),packets,delay,double(stats.nTotalDroppedPackets),
                     double(stats.nTotalMarkedPackets),utilization,minimum,maximum});
 if (Simulator::Now().GetSeconds()+s->interval<s->duration)
  Simulator::Schedule(Seconds(s->interval),&Control,s);
}
std::string Run(int mode,uint32_t seed,double minimum,double maximum,double probability,uint32_t limit,
 bool ecn,double gain,double target,double duration,double interval,double bottleneckMbps,
 double bottleDelayMs,double accessMbps,double accessDelayMs,uint32_t flows,uint32_t packetBytes,
 double appStart,double appStop,double histogramBin) {
 RngSeedManager::SetSeed(seed); RngSeedManager::SetRun(1);
 Config::SetDefault("ns3::TcpL4Protocol::SocketType",TypeIdValue(TcpNewReno::GetTypeId()));
 Config::SetDefault("ns3::TcpSocketBase::UseEcn",StringValue(ecn?"On":"Off"));
 Config::SetDefault("ns3::TcpSocket::SegmentSize",UintegerValue(packetBytes));
 NodeContainer routers; routers.Create(2);
 NodeContainer senders,receivers; senders.Create(flows); receivers.Create(flows);
 InternetStackHelper stack; stack.Install(routers); stack.Install(senders); stack.Install(receivers);
 PointToPointHelper access,bottleneck;
 access.SetDeviceAttribute("DataRate",DataRateValue(DataRate(uint64_t(accessMbps*1e6))));
 access.SetChannelAttribute("Delay",TimeValue(MilliSeconds(accessDelayMs)));
 bottleneck.SetDeviceAttribute("DataRate",DataRateValue(DataRate(uint64_t(bottleneckMbps*1e6))));
 bottleneck.SetChannelAttribute("Delay",TimeValue(MilliSeconds(bottleDelayMs)));
 bottleneck.SetQueue("ns3::DropTailQueue<Packet>","MaxSize",QueueSizeValue(QueueSize("1p")));
 NetDeviceContainer central=bottleneck.Install(routers);
 TrafficControlHelper traffic;
 traffic.SetRootQueueDisc("ns3::RedQueueDisc","MinTh",DoubleValue(minimum),"MaxTh",DoubleValue(maximum),
  "LInterm",DoubleValue(1.0/probability),"MaxSize",QueueSizeValue(QueueSize(std::to_string(limit)+"p")),
  "UseEcn",BooleanValue(ecn),"UseHardDrop",BooleanValue(false),"MeanPktSize",UintegerValue(packetBytes),
  "LinkBandwidth",DataRateValue(DataRate(uint64_t(bottleneckMbps*1e6))),"LinkDelay",TimeValue(MilliSeconds(bottleDelayMs)));
 auto queues=traffic.Install(central);
 DynamicCast<RedQueueDisc>(queues.Get(0))->AssignStreams(1);
 DynamicCast<RedQueueDisc>(queues.Get(1))->AssignStreams(2);
 Ipv4AddressHelper address; address.SetBase("10.0.0.0","255.255.255.0"); address.Assign(central);
 ApplicationContainer sinkApps;
 for (uint32_t i=0;i<flows;++i) {
  NodeContainer left(senders.Get(i),routers.Get(0)),right(routers.Get(1),receivers.Get(i));
  auto dl=access.Install(left),dr=access.Install(right);
  std::string l="10.1."+std::to_string(i)+".0",r="10.2."+std::to_string(i)+".0";
  address.SetBase(l.c_str(),"255.255.255.0"); address.Assign(dl);
  address.SetBase(r.c_str(),"255.255.255.0"); auto ip=address.Assign(dr);
  uint16_t port=9000+i;
  PacketSinkHelper sink("ns3::TcpSocketFactory",InetSocketAddress(Ipv4Address::GetAny(),port));
  sinkApps.Add(sink.Install(receivers.Get(i)));
  ApplicationContainer source;
  if (mode==0 || (mode==2 && i<flows/2)) {
   BulkSendHelper bulk("ns3::TcpSocketFactory",InetSocketAddress(ip.GetAddress(1),port));
   bulk.SetAttribute("MaxBytes",UintegerValue(0)); bulk.SetAttribute("SendSize",UintegerValue(packetBytes));
   source=bulk.Install(senders.Get(i));
  } else {
   OnOffHelper burst("ns3::TcpSocketFactory",InetSocketAddress(ip.GetAddress(1),port));
   burst.SetAttribute("DataRate",DataRateValue(DataRate("20Mbps")));
   burst.SetAttribute("PacketSize",UintegerValue(packetBytes));
   burst.SetAttribute("OnTime",StringValue("ns3::ExponentialRandomVariable[Mean=0.15]"));
   burst.SetAttribute("OffTime",StringValue("ns3::ExponentialRandomVariable[Mean=0.30]"));
   source=burst.Install(senders.Get(i)); burst.AssignStreams(senders.Get(i),100+i*2);
  }
  source.Start(Seconds(appStart+0.01*i)); source.Stop(Seconds(appStop));
 }
 sinkApps.Start(Seconds(0)); sinkApps.Stop(Seconds(duration));
 Ipv4GlobalRoutingHelper::PopulateRoutingTables();
 FlowMonitorHelper flow;
 flow.SetMonitorAttribute("DelayBinWidth",DoubleValue(histogramBin));
 auto monitor=flow.InstallAll(); auto classifier=DynamicCast<Ipv4FlowClassifier>(flow.GetClassifier());
 ControlState state{DynamicCast<RedQueueDisc>(queues.Get(0)),monitor,minimum,maximum,gain,target,interval,duration,bottleneckMbps*1e6,limit};
 Simulator::Schedule(Seconds(0),&Control,&state);
 Simulator::Stop(Seconds(duration)); Simulator::Run(); monitor->CheckForLostPackets(Seconds(0));
 uint64_t tx=0,rx=0,lost=0,totalPayload=0; std::vector<uint64_t> histogram;
 double sum=0,square=0;
 for(uint32_t i=0;i<flows;++i) {
  uint64_t bytes=DynamicCast<PacketSink>(sinkApps.Get(i))->GetTotalRx();
  totalPayload+=bytes;sum+=bytes;square+=double(bytes)*bytes;
 }
 for(const auto& entry:monitor->GetFlowStats()) {
  auto tuple=classifier->FindFlow(entry.first);
  if(tuple.destinationPort<9000 || tuple.destinationPort>=9000+flows) continue;
  const auto& stats=entry.second;tx+=stats.txPackets;rx+=stats.rxPackets;lost+=stats.lostPackets;
  for(uint32_t i=0;i<stats.delayHistogram.GetNBins();++i) {
   if(histogram.size()<=i) histogram.resize(i+1,0);
   histogram[i]+=stats.delayHistogram.GetBinCount(i);
  }
 }
 double p95=0;uint64_t cumulative=0;
 for(size_t i=0;i<histogram.size();++i) {cumulative+=histogram[i];if(cumulative>=std::ceil(0.95*rx)){p95=(i+1)*histogramBin;break;}}
 double goodput=totalPayload*8.0/((appStop-appStart)*bottleneckMbps*1e6);
 double fairness=square ? sum*sum/(flows*square) : 0;
 std::ostringstream out;out<<std::setprecision(17);
 out<<"{\"normalized_goodput\":"<<std::min(1.0,goodput)<<",\"delay_p95_seconds\":"<<p95
    <<",\"loss_rate\":"<<(tx ? double(lost)/tx : 0)<<",\"jain_fairness\":"<<fairness
    <<",\"tx_packets\":"<<tx<<",\"rx_packets\":"<<rx<<",\"lost_packets\":"<<lost
    <<",\"payload_bytes\":"<<totalPayload<<",\"control_trace\":[";
 for(size_t i=0;i<state.trace.size();++i){if(i)out<<",";out<<"[";for(size_t j=0;j<state.trace[i].size();++j){if(j)out<<",";out<<state.trace[i][j];}out<<"]";}
 out<<"]}";
 Simulator::Destroy();return out.str();
}
}

#ifdef SPECRSI_AQM_STANDALONE
#include <iostream>
int main(int argc,char** argv) {
 if(argc!=21) {std::cerr<<"Expected 20 native scalar arguments";return 2;}
 try {
  std::cout<<specrsi_native_aqm::Run(std::stoi(argv[1]),std::stoul(argv[2]),std::stod(argv[3]),std::stod(argv[4]),
   std::stod(argv[5]),std::stoul(argv[6]),std::stoi(argv[7]),std::stod(argv[8]),std::stod(argv[9]),
   std::stod(argv[10]),std::stod(argv[11]),std::stod(argv[12]),std::stod(argv[13]),std::stod(argv[14]),
   std::stod(argv[15]),std::stoul(argv[16]),std::stoul(argv[17]),std::stod(argv[18]),std::stod(argv[19]),std::stod(argv[20]))<<std::endl;
 } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
 return 0;
}
#endif
