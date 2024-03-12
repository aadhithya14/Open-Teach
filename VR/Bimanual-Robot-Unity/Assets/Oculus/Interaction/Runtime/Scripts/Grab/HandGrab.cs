/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using Oculus.Interaction.GrabAPI;
using Oculus.Interaction.Input;

namespace Oculus.Interaction.Grab
{
    public enum GrabType
    {
        PinchGrab = 1 << 0,
        PalmGrab = 1 << 1
    }

    public interface IHandGrabber
    {
        HandGrabAPI HandGrabApi { get; }
        GrabTypeFlags SupportedGrabTypes { get; }
        IHandGrabbable TargetInteractable { get; }
    }

    public interface IHandGrabbable
    {
        GrabTypeFlags SupportedGrabTypes { get; }
        GrabbingRule PinchGrabRules { get; }
        GrabbingRule PalmGrabRules { get; }
    }

    public class HandGrabbableData : IHandGrabbable
    {
        public GrabTypeFlags SupportedGrabTypes { get; set; } = GrabTypeFlags.All;
        public GrabbingRule PinchGrabRules { get; set; } = GrabbingRule.DefaultPinchRule;
        public GrabbingRule PalmGrabRules { get; set; } = GrabbingRule.DefaultPalmRule;
    }

    public static class HandGrab
    {
        public static void StoreGrabData(IHandGrabber grabber,
            IHandGrabbable grabbable, ref HandGrabbableData cache)
        {
            HandGrabAPI api = grabber.HandGrabApi;

            cache.SupportedGrabTypes = GrabTypeFlags.None;

            if (SupportsPinch(grabber, grabbable))
            {
                HandFingerFlags pinchFingers = api.HandPinchGrabbingFingers();
                if (api.IsSustainingGrab(grabbable.PinchGrabRules, pinchFingers))
                {
                    cache.SupportedGrabTypes |= GrabTypeFlags.Pinch;
                    cache.PinchGrabRules = new GrabbingRule(pinchFingers, grabbable.PinchGrabRules);
                }
            }
            if (SupportsPalm(grabber, grabbable))
            {
                HandFingerFlags palmFingers = api.HandPalmGrabbingFingers();
                if (api.IsSustainingGrab(grabbable.PalmGrabRules, palmFingers))
                {
                    cache.SupportedGrabTypes |= GrabTypeFlags.Palm;
                    cache.PalmGrabRules = new GrabbingRule(palmFingers, grabbable.PalmGrabRules);
                }
            }
        }

        public static float ComputeHandGrabScore(IHandGrabber grabber,
            IHandGrabbable grabbable, out GrabTypeFlags handGrabTypes)
        {
            HandGrabAPI api = grabber.HandGrabApi;
            handGrabTypes = GrabTypeFlags.None;
            float handGrabScore = 0f;

            if (SupportsPinch(grabber, grabbable))
            {
                float pinchStrength = api.GetHandPinchScore(grabbable.PinchGrabRules, false);
                if (pinchStrength > handGrabScore)
                {
                    handGrabScore = pinchStrength;
                    handGrabTypes = GrabTypeFlags.Pinch;
                }
            }

            if (SupportsPalm(grabber, grabbable))
            {
                float palmStrength = api.GetHandPalmScore(grabbable.PalmGrabRules, false);
                if (palmStrength > handGrabScore)
                {
                    handGrabScore = palmStrength;
                    handGrabTypes = GrabTypeFlags.Palm;
                }
            }

            return handGrabScore;
        }

        public static bool ComputeShouldSelect(IHandGrabber grabber,
            IHandGrabbable grabbable, out GrabTypeFlags selectingGrabTypes)
        {
            if (grabbable == null)
            {
                selectingGrabTypes = GrabTypeFlags.None;
                return false;
            }

            HandGrabAPI api = grabber.HandGrabApi;
            selectingGrabTypes = GrabTypeFlags.None;
            if (SupportsPinch(grabber, grabbable) &&
                 api.IsHandSelectPinchFingersChanged(grabbable.PinchGrabRules))
            {
                selectingGrabTypes |= GrabTypeFlags.Pinch;
            }

            if (SupportsPalm(grabber, grabbable) &&
                 api.IsHandSelectPalmFingersChanged(grabbable.PalmGrabRules))
            {
                selectingGrabTypes |= GrabTypeFlags.Palm;
            }

            return selectingGrabTypes != GrabTypeFlags.None;
        }

        public static bool ComputeShouldUnselect(IHandGrabber grabber,
            IHandGrabbable grabbable)
        {
            HandGrabAPI api = grabber.HandGrabApi;
            HandFingerFlags pinchFingers = api.HandPinchGrabbingFingers();
            HandFingerFlags palmFingers = api.HandPalmGrabbingFingers();

            if (grabbable.SupportedGrabTypes == GrabTypeFlags.None)
            {
                if (!api.IsSustainingGrab(GrabbingRule.FullGrab, pinchFingers) &&
                    !api.IsSustainingGrab(GrabbingRule.FullGrab, palmFingers))
                {
                    return true;
                }
                return false;
            }

            bool pinchHolding = false;
            bool palmHolding = false;
            bool pinchReleased = false;
            bool palmReleased = false;

            if (SupportsPinch(grabber, grabbable.SupportedGrabTypes))
            {
                pinchHolding = api.IsSustainingGrab(grabbable.PinchGrabRules, pinchFingers);
                if (api.IsHandUnselectPinchFingersChanged(grabbable.PinchGrabRules))
                {
                    pinchReleased = true;
                }
            }

            if (SupportsPalm(grabber, grabbable.SupportedGrabTypes))
            {
                palmHolding = api.IsSustainingGrab(grabbable.PalmGrabRules, palmFingers);
                if (api.IsHandUnselectPalmFingersChanged(grabbable.PalmGrabRules))
                {
                    palmReleased = true;
                }
            }

            return !pinchHolding && !palmHolding && (pinchReleased || palmReleased);
        }

        public static HandFingerFlags GrabbingFingers(IHandGrabber grabber,
            IHandGrabbable grabbable)
        {
            HandGrabAPI api = grabber.HandGrabApi;
            if (grabbable == null)
            {
                return HandFingerFlags.None;
            }

            HandFingerFlags fingers = HandFingerFlags.None;

            if (SupportsPinch(grabber, grabbable))
            {
                HandFingerFlags pinchingFingers = api.HandPinchGrabbingFingers();
                grabbable.PinchGrabRules.StripIrrelevant(ref pinchingFingers);
                fingers = fingers | pinchingFingers;
            }

            if (SupportsPalm(grabber, grabbable))
            {
                HandFingerFlags grabbingFingers = api.HandPalmGrabbingFingers();
                grabbable.PalmGrabRules.StripIrrelevant(ref grabbingFingers);
                fingers = fingers | grabbingFingers;
            }

            return fingers;
        }

        private static bool SupportsPinch(IHandGrabber grabber,
            IHandGrabbable grabbable)
        {
            return SupportsPinch(grabber, grabbable.SupportedGrabTypes);
        }

        private static bool SupportsPalm(IHandGrabber grabber,
            IHandGrabbable grabbable)
        {
            return SupportsPalm(grabber, grabbable.SupportedGrabTypes);
        }

        private static bool SupportsPinch(IHandGrabber grabber,
            GrabTypeFlags grabTypes)
        {
            return (grabber.SupportedGrabTypes & GrabTypeFlags.Pinch) != 0 &&
                (grabTypes & GrabTypeFlags.Pinch) != 0;
        }

        private static bool SupportsPalm(IHandGrabber grabber,
            GrabTypeFlags grabTypes)
        {
            return (grabber.SupportedGrabTypes & GrabTypeFlags.Palm) != 0 &&
                (grabTypes & GrabTypeFlags.Palm) != 0;
        }
    }
}
