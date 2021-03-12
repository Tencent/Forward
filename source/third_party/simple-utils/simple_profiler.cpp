//////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                              //
// Copyright (c) 2019 Tencent.com, Inc. All Rights Reserved                                     //
// Author: Aster Jian (asterjian@qq.com)                                                        //
//                                                                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////
#include "simple_profiler.h"
#include "simple_table.h"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <unordered_map>

namespace utils
{
    static float GetTimeMs()
    {
        using clock = std::chrono::high_resolution_clock;        
        static clock::time_point start_ = clock::now();

        using duration = std::chrono::duration<float, std::milli>;
        return duration(clock::now() - start_).count();
    }

    struct Scope
    {
        Scope(Scope* parent) : parent_(parent)
        {
        }

        ~Scope()
        {
            for (auto& iter : children_)
            {
                delete iter.second;
            }
        }

        void Enter(float enter_time)
        {
            time_ = enter_time;
        }

        void Leave(float leave_time)
        {
            time_ = leave_time - time_;

            total_time_ += time_;
            total_count_ += 1;

            min_time_ = std::min(min_time_, time_);
            max_time_ = std::max(max_time_, time_);
        }

        Scope* GetChild(const char* name)
        {
            auto position = children_.find(name);
            if (position != children_.end())
            {
                return position->second;
            }

            Scope* child = new Scope(this);
            children_[name] = child;

            return child;
        }

        void Print(Table& table, const char* name, float parent_time, const std::string& prefix)
        {
            if (parent_time == 0.0f)
            {
                parent_time = total_time_;
            }

            table.AddRow().AddString(prefix + name)
                .AddInt(total_count_)
                .AddFloat(total_time_)
                .AddFloat(total_time_ / total_count_)
                .AddFloat(min_time_)
                .AddFloat(max_time_)
                .AddFloat(total_time_ / parent_time);
            
            for (auto& iter : children_)
            {
                iter.second->Print(table, iter.first, total_time_, prefix + "    ");
            }
        }

        Scope* parent_{ nullptr };
        std::unordered_map<const char*, Scope*> children_;

        float time_{ 0.0f };
        int total_count_{ 0 };
        float total_time_{ 0.0f };
        float min_time_{ std::numeric_limits<float>::max() };
        float max_time_{ 0.0f };
    };

    thread_local Profiler* Profiler::s_instance = nullptr;

    Profiler::Profiler(const char* name) : name_(name)
    {
        s_instance = this;

        root_ = new Scope(nullptr);
        root_->Enter(GetTimeMs());

        current_ = root_;
    }

    Profiler::~Profiler()
    {
        delete root_;

        s_instance = nullptr;
    }

    void Profiler::EnterScope(const char* name)
    {
        current_ = current_->GetChild(name);

        current_->Enter(GetTimeMs());
    }

    void Profiler::LeaveScope()
    {
        current_->Leave(GetTimeMs());
        current_ = current_->parent_;
    }

    std::mutex g_print_mutex;

    void Profiler::Print(std::ostream& os)
    {
        std::lock_guard<std::mutex> guard(g_print_mutex);

        root_->Leave(GetTimeMs());

        Table table;

        table.AddColumn("Name");
        table.AddColumn("Count");
        table.AddColumn("Total(ms)");
        table.AddColumn("Average(ms)");
        table.AddColumn("Min(ms)");
        table.AddColumn("Max(ms)");
        table.AddColumn("Percentage").SetPrecision(2).SetPercentage(true);

        root_->Print(table, name_ ? name_ : "Root", 0LL, "");

        table.Print(os);

        root_->Enter(GetTimeMs());
    }
}
